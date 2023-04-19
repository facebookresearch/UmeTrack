"""
This module defines methods for running asyncio code from non-asyncio code.
"""

import asyncio
import concurrent
import contextlib
import functools
import logging
import os
import threading
from typing import Awaitable, Iterable, TypeVar

from . import async_utils as au


T = TypeVar("T")
logger = logging.getLogger(__name__)


_async_thread = None


def _async_main(loop):
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def get_event_loop():
    global _async_thread
    th = _async_thread
    if th is not None and th.is_alive():
        return th.loop

    if th is None:
        logger.info(f"creating event loop thread for {os.getpid()}")
    elif th.original_pid == os.getpid():
        # thread died somehow? don't recreate it.
        raise RuntimeError("event loop thread is dead")
    else:
        # thread not running because we're in a forked process
        logger.info(
            f"recreating event loop thread for {os.getpid()} after fork from {th.original_pid}"
        )

    loop = asyncio.new_event_loop()
    th = threading.Thread(
        target=_async_main, name="s2_async", args=(loop,), daemon=True
    )
    th.original_pid = os.getpid()
    th.loop = loop

    _async_thread = th
    th.start()
    return loop


def shutdown_event_loop():
    """
    Shut down the main event loop.

    This is needed for unit tests that so they don't leave threads
    running, so that a later multiprocess fork in an unrelated
    test doesn't hang.

    Not needed in other code because other code can use `forkserver`
    or `spawn` for multiprocessing.
    """
    global _async_thread
    th = _async_thread
    if th is not None:
        if th.is_alive():
            loop = th.loop
            logger.info(f"shutting down event loop thread for {os.getpid()}")
            loop.call_soon_threadsafe(loop.stop)
            th.join()
        _async_thread = None


def spawn(awaitable):
    return au.run_coroutine_threadsafe(awaitable, get_event_loop())


def run(awaitable: Awaitable[T]) -> T:
    """
    Run the given awaitable on the thread event loop.

    Similar to `asyncio.run()`, except it talks to an event loop on a worker
    thread instead of running one inline, so the event loop can do actual
    concurrent work. Unlike `asyncio.run()` it explicitly does not clean up
    incomplete generators before returning, so you can continue to iterate them
    in a subsequent call.
    """
    return spawn(awaitable).result()


class _AsyncToSyncIterator:
    """
    (Internal) Converts async to a sync iterator

    Simple wrapper that abstracts away the small amount of glue required to
    switch from an async iteration to sync.
    """

    __slots__ = ("_iter", "_closed")

    def __init__(self, async_iter):
        self._iter = async_iter

    def __iter__(self):
        self._iter = self._iter.__aiter__()
        return self

    def __next__(self):
        try:
            x = run(self._iter.__anext__())
            return x
        except StopAsyncIteration as e:
            raise StopIteration from e

    def send(self, val):
        try:
            return run(self._iter.asend(val))
        except StopAsyncIteration as e:
            raise StopIteration from e

    def close(self):
        i = self._iter
        if i:
            return run(i.aclose())


def async_to_sync_iterator(async_iter):
    """
    Turn an async iterator into a normal iterator via `nested_async.run`:func:

    Note that (like all Python iterators), to ensure cleanup when the loop
    doesn't finish, it also needs a `with` block. This class returns a context
    manager that must be entered so that this step doesn't get missed.

    Example
    -------

    >>> async def aiter():
    ...     for i in range(10):
    ...         await asyncio.sleep(1)
    ...         yield i
    >>> with async_to_sync_iterator(aiter()) as it:
    ...     list(it)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


    >>> async def aiter(n):
    ...     try:
    ...         for i in range(n):
    ...             await asyncio.sleep(1)
    ...             yield i
    ...     finally:
    ...         print('essentional cleanup!')
    >>> with async_to_sync_iterator(aiter(n)) as it:
    ...     list(it)
    essential cleanup!
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return contextlib.closing(_AsyncToSyncIterator(async_iter))


def wrap(f):
    """
    decorator that wraps an async function with `run()`.

    Example::

        @nested_async.wrap
        async def afun():
            await asyncio.sleep(1)
            return 10

        def sfun():  # note -- not async!
            print(afun())
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return run(f(*args, **kwargs))

    return wrapper


async def _put_prefetch_tasks(seq, channel):
    def decrement_inflight_count(_):
        channel.count_inflight -= 1

    async with au.TaskGroup(cancel_on_error=False) as taskgroup:
        # Iterate over the sequence, starting an async task for each element.
        # The bounded channel limits the number of tasks we'll have in flight.
        with channel, contextlib.closing(seq) as it:
            try:
                for awaitable in it:
                    task = taskgroup.create_task(awaitable)
                    channel.count_inflight += 1
                    task.add_done_callback(decrement_inflight_count)
                    await channel.put(
                        au.run_coroutine_threadsafe(task, task.get_loop())
                    )
            except au.ChannelClosedError as e:
                logger.exception(f"Caught ChannelClosedError: {e}")
                taskgroup.cancel()
                ...


def prefetch_sequence(seq: Iterable[Awaitable[T]], prefetch_size: int) -> Iterable[T]:
    """
    Converts an iterable of Awaitable[T] to an iterable of T

    Runs an event loop on a worker thread to process `prefetch_size`
    elements in parallel.

    Note: this takes a regular (non-async) iterable of awaitables, not
          an "async iterable".
    """
    channel = au.AsyncToSyncChannel(maxsize=prefetch_size)
    channel.count_inflight = 0
    prefetch_fut = spawn(_put_prefetch_tasks(seq, channel))
    try:
        with channel:
            for f in channel:
                y = f.result()
                yield y
                del y
                del f

    except GeneratorExit as e:
        logger.exception(f"Caught GeneratorExit exception: {e}")
        # Our caller did a 'break' from our sequence.
        # Discard any pending work. If that produces an error other than
        # cancellation, that error will be re-raised.
        prefetch_fut.cancel()
        with contextlib.suppress(concurrent.futures.CancelledError):
            prefetch_fut.result()
        raise
    except BaseException as e:
        logger.exception(f"Caught BaseException exception: {e}")
        # discard errors during cancellation in favor of the
        # one that got us here.
        prefetch_fut.cancel()
        concurrent.futures.wait([prefetch_fut])
        raise
    prefetch_fut.result()
