"""
This module contains utilities for working with asyncio in Python.

## Replacements for broken asyncio functions

These functions here have the same name and general purpose as existing
functions in asyncio, but with specific fixes related to error handling and
cleanup.

This is to work around design flaws in asyncio, which make it extremely hard to
do correct error handling and cleanup. It's very easy to silently drop errors in
asyncio, and very easy to cause hangs.

This module borrows ideas from `trio` (which is a vastly improved asyncio
replacement for Python), but still tries to work with the asyncio model.

.. autofunction:: run_in_executor
.. autofunction:: run_coroutine_threadsafe
.. autofunction:: run_coroutine_threadsafe

## Missing asyncio functionality

These are classes which make it easier to reliably handle errors and cleanup in
async code. :class:`aclosing` is required for determinstic cleanup of async
generators, and :class:`TaskGroup` enables structured concurrency.

.. autoclass:: aclosing
.. autoclass:: TaskGroup
.. autoclass:: AsyncToSyncChannel
"""
import asyncio
import collections
import concurrent.futures
import functools
import logging
import sys
import threading
import time
import weakref
from typing import Awaitable, Generic, Iterator, TypeVar


logger = logging.getLogger(__name__)


T = TypeVar("T")


async def run_in_executor(executor, func, *args):
    """
    Awaitable that runs `func` in the given executor::

        def slow_func(...):
            ...

        await async_utils.run_in_executor(my_executor, slow_func,

    Defined because the loop should be implicit in this call; there's no
    possible correct value you can pass except the running loop.

    Parameters
    ----------
    executor : concurrent.futures.Executor

    func Function to run
    """
    return await asyncio.get_running_loop().run_in_executor(executor, func, *args)


def run_coroutine_threadsafe(
    coro_or_task: Awaitable[T], loop: asyncio.AbstractEventLoop
) -> concurrent.futures.Future:
    """
    Like :func:`asyncio.run_coroutine_threadsafe` but cancellation works properly.

    The problem:

    :meth:`concurrent.futures.Future.cancel` sets the future to a "done" state
    *immediately*. It provides no way whatsoever to wait for cleanup to complete,
    nor to report errors that occurred during cleanup.

    :meth:`asyncio.run_coroutine_threadsafe` returns a Future so it inherits this
    problem. This leads to race conditions in cleanup code and makes it nearly
    impossible to robustly shut things down.

    This function returns a derived `Future` that overrides `cancel()` to request
    the task be cancelled, but doesn't mark the Future as done until the task has
    actually finished cancellation.

    In other words, if you do::

        state = "initial"
        async def my_async_fn():
            global did_cleanup
            try:
                state = "dirty"
                await asyncio.sleep(100)
            finally:
                state = "clean"

        f = run_coroutine_threadsafe(my_async_fn(), loop)
        ...
        f.cancel()
        concurrent.futures.wait([f])
        assert state in ("initial", "clean")

    If the coroutine made it into the 'try' block, then the `wait()` call
    (or fetching `result()` or `exception()`, which implicitly wait) will
    not return until the "finally" block has run too.

    Unlike the asyncio version, you can also pass an existing `Task` object
    instead of a coroutine, so that callers can control task creation if
    necessary.

    See also
    --------
    :meth:`nested_async.spawn`, which just calls this function using the
    default nested_async loop.
    """
    return _AFuture(coro_or_task, loop)


class _AFuture(concurrent.futures.Future):
    """
    (Internal) Future type with cleaner cancel semantics.

    Allows for tasks to actually cancel when a future is cancelled.  This is as
    opposed to immediately reporting the task done without any possibility of
    cleanup.
    """

    def __init__(self, coro_or_task, loop):
        if not (
            asyncio.iscoroutine(coro_or_task) or isinstance(coro_or_task, asyncio.Task)
        ):
            raise TypeError("A coroutine or task object is required")

        super().__init__()
        self._loop = loop
        self._task = None
        self._cancelled = False

        if isinstance(coro_or_task, asyncio.Task):
            self._task = coro_or_task
            coro_or_task.add_done_callback(self._set_state)
        else:
            loop.call_soon_threadsafe(self._create_task_cb, coro_or_task)

    def cancel(self):
        # Cancel the async task associated with this future.
        # This does *not* call super.cancel(), because
        # concurrent.futures.Future makes it impossible to wait for
        # a result once that has been done. Instead it schedules
        # an async cancellation and then lets that propagate normally.
        self._loop.call_soon_threadsafe(self._cancel_cb)
        return True

    def _create_task_cb(self, coro):
        loop = self._loop
        self._task = task = loop.create_task(coro)
        task.add_done_callback(self._set_state)
        if self._cancelled:
            task.cancel()

    def _cancel_cb(self):
        if not self._cancelled:
            self._cancelled = True
            if self._task:
                self._task.cancel()

    def _set_state(self, task):
        assert task.done()
        if task.cancelled():
            super().cancel()
        if not self.set_running_or_notify_cancel():
            return
        exc = task.exception()
        if exc is not None:
            # convert to concurrent.futures.CancelledError to match concurrent
            # API. We don't have to convert TimeoutError because it will
            # already be the concurrent.future version.
            if isinstance(exc, asyncio.CancelledError):
                exc = type(exc)(*exc.args)
            self.set_exception(exc)
        else:
            self.set_result(task.result())


class aclosing:  # pylint: disable=invalid-name, well-known terminology
    """
    Context manager for async iterables.

    Async iteration in Python is broken. To clean up correctly in case
    of exceptions (or even `break` statements), *every* loop over *any*
    async generator must be wrapped in an aclosing context manager:

        async with aclosing(my_async_gen()) as it:
            async for x in it:
                do_stuff_with(x)

    This is also true of non-async generators, although since they
    are much less likely to need cleanup, it isn't a big problem in
    practice. Async generators *always* need cleanup, so must *always*
    be wrapped in an `aclosing` context manager.

    Yet Python doesn't provide such a context manager.

    https://www.python.org/dev/peps/pep-0533/ cannot come too soon.
    """

    def __init__(self, aclosable):
        self._obj = aclosable

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *exn):
        await self._obj.aclose()


async def wait(fs, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
    """Same as asyncio.wait but allows empty list of futures."""
    if fs:
        return await asyncio.wait(fs, timeout=timeout, return_when=return_when)
    else:
        return (set(), set())


class TaskGroup:
    """
    An async context manager that waits for child tasks on exit.

    When writing parallel async code (i.e. using create_task to actually run
    tasks in parallel), it can be difficult to ensure everything is cleaned up
    if one task fails with an error.

    This context manager keeps track of a set of tasks that should all be
    complete by the time the group context exits. If the context exits normally,
    it waits for all the tasks to complete. If it exits via an exception, it
    cancels all tasks and then waits for them.

    Tasks can be added to the context manager explicitly via :meth:`add`, or
    (more easily) by using :meth:`create_task` in place of
    :func:`asyncio.create_task`

    Example::

        async def read_blob(name): ...

        async def concat_blobs(name1, name2):
            async with async_utils.TaskGroup() as tgroup:
                b1 = tgroup.create_task(read_blob(name1))
                b2 = tgroup.create_task(read_blob(name2))
                return await b1 + await b2

    If the context exits via an exception, any still-running tasks in the group
    will be cancelled (unless cancel_on_error=False). Any additional errors
    thrown during cleanup will be discarded.

    On normal exit, it will wait for all tasks to complete. If any of them raise
    an error, the remaining tasks will be cancelled.

    This means there's a "join" at the end of the with block, so the fact that
    `concat_blobs` contains concurrency is irrelevant to its callers, enabling
    compostion of the code.

    There's an exellent explanation of this model at
    https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/
    with :class:`TaskGroup` being equivalent to Trio's `nursery`, and
    `group.create_task()` equivalent to `nursery.start_soon()`.
    """

    def __init__(self, cancel_on_error=True):
        self.tasks = weakref.WeakSet()
        self.cancelled = False
        self.exceptions = []
        self.context_task = None
        self._cancel_on_error = cancel_on_error

    def add(self, task: asyncio.Task) -> asyncio.Task:
        """
        Add an existing Task to this manager.
        """
        if self.cancelled:
            raise ValueError("attempting to add task to cancelled taskgroup")
        self.tasks.add(task)
        return task

    if sys.version_info >= (3, 8):
        # 'name' param added in 3.8
        def create_task(self, awaitable, name=None) -> asyncio.Task:
            """
            Start a task managed by this TaskGroup.
            """
            if self.cancelled:
                raise ValueError("attempting to create task in cancelled taskgroup")
            return self.add(asyncio.create_task(awaitable, name=name))

    else:

        def create_task(self, awaitable, name=None) -> asyncio.Task:
            """
            Start a task managed by this TaskGroup.
            """
            del name
            if self.cancelled:
                raise ValueError("attempting to create task in cancelled taskgroup")
            return self.add(asyncio.create_task(awaitable))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            if self._cancel_on_error or isinstance(exc, asyncio.CancelledError):
                self.cancel()

        # wait for all pending tasks to finish
        cleanup_error = None
        while self.tasks:
            try:
                done, _ = await wait({*self.tasks}, return_when=asyncio.FIRST_EXCEPTION)
                self.tasks -= done
            except asyncio.CancelledError as c:
                # the context task was cancelled before `wait` completed.
                # Child tasks may not be done yet, so we have to wait for
                # them again, but we'll no longer exit with success.
                self.cancel()
                cleanup_error = c

            # save and reraise a non-cancellation exception if we hit one,
            # so we don't lose track of real errors.
            if exc_type is None and cleanup_error is None:
                for t in done:
                    if not t.cancelled() and t.exception():
                        cleanup_error = t.exception()
                        self.cancel()
                        break

        if exc_type is None and cleanup_error:
            raise cleanup_error

    def cancel(self):
        """
        Cancel all tasks in the group.

        `cancel()` must be idempotent to avoid double-cancelling child tasks;
        doing so would raise CancelledError in them twice and interrupt any
        cleanup they may have been doing.
        """
        if not self.cancelled:
            self.cancelled = True
            not_done = {t for t in self.tasks if not t.done()}
            for t in not_done:
                t.cancel()


class ChannelClosedError(ValueError):
    ...


class AsyncToSyncChannel(Generic[T]):
    """
    Bounded MPMC channel with async writer and sync reader.

    Why this is needed
    ------------------

    `queue.Queue` and `queue.SimpleQueue` provide queues between threads.
    `asyncio.Queue` provides queues between async tasks.

    However, there is no bounded queue type for sending data between async tasks
    and threads. This class provides that for the async->sync direction.

    Cleanup
    -------

    Both the reader and writer should wrap the queue in a 'with' block::

        async def writer(queue):
            with queue:
                for x in range(100): queue.put(x)

        def reader(queue):
            with queue:
                for x in queue: print(x)

    This will call `close()` when the block exits, which ensures that the other
    end will not hang forever waiting for an item that won't arrive.

    If the reader exits first, the writer will fail with `ChannelClosedError`
    when it tries to put to a closed queue. If the writer exits first, the
    reader will first process the remaining items normally, then get
    `StopIteration` (i.e. simply exit the for-loop normally).
    """

    def __init__(self, maxsize):
        self._queue = collections.deque()
        self._lock = threading.Lock()
        self._nonempty_event = threading.Event()
        self._closed = False
        self._maxsize = maxsize
        self._put_waiter = None

    async def put(self, obj: T):
        """
        Put an item onto the queue.

        Blocks if the queue is full. Raises ValueError if the queue is closed.
        """
        while True:
            with self._lock:
                if self._closed:
                    raise ChannelClosedError("attempt to push to closed queue")
                if len(self._queue) < self._maxsize:
                    self._queue.append(obj)
                    self._nonempty_event.set()
                    return True
                else:
                    f = self._put_waiter
                    if f is None:
                        self._put_waiter = f = asyncio.Future()
            await f

    def get(self) -> T:
        while True:
            with self._lock:
                if len(self._queue) > 0:
                    # wake up writer since we're making some room
                    self._awaken_writers()
                    return self._queue.popleft()
                elif self._closed:
                    raise StopIteration
                else:
                    self._nonempty_event.clear()
            self._nonempty_event.wait()

    def close(self, discard=False):
        """
        Mark the queue as closed.

        Attempting to `put` to a closed queue raises ValueError.

        Attempting to iterate a closed queue will continue to yield any remaining
        items (unless discard=True) and then raise `StopIteration`.

        Parameters
        ----------
        discard : bool
            If True, any pending items in the queue will be discarded.
        """
        with self._lock:
            self._closed = True
            if discard:
                self._queue.clear()
            self._nonempty_event.set()
            self._awaken_writers()

    def closed(self) -> bool:
        return self._closed

    # implement iterable interface for consuming queue
    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return self.get()

    # implement context manager interface for closing queue
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _awaken_writers(self):
        # awaken async task waiting for _put_waiter.
        # this can only be called when _lock is held!
        assert self._lock.locked()
        if self._put_waiter:
            self._put_waiter.get_loop().call_soon_threadsafe(
                self._put_waiter.set_result, None
            )
            self._put_waiter = None
