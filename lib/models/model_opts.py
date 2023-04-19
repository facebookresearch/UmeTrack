from dataclasses import dataclass


@dataclass
class ModelOpts:
    network: str = "resnet_layers_2352-f32"
    # #channels of feature map after backbone (that would be used for fusion)
    nImageFeatureChannels: int = 72
    # #channels of skeleton feature map that will be concatenated with temporal
    # feature map
    nSkeletonFeatureChannels: int = 4
    # #channels of temporal memory (as the hidden state in temporal model)
    nTemporalMemoryChannels: int = 18
    # In the multi-view fusion stage, the canonical space is identical to the
    # cam0 space if `useUnscaledAsCanonical=True`. The final fused features
    # will be in the coordinate system of the canonical space.
    useUnscaledAsCanonical: bool = False
    # #multi-view blocks
    nMultiViewFusionBlocks: int = 2
    # #temporal_blocks (conv + relu)
    nTemporalBlocks: int = 3
    # #resnet_blocks in pose regressor
    nPoseRegressionBlocks: int = 2
    # ratio of the n_channel for spatial FTL (i.e., we won't apply
    # transformation for the rest of (1-ratio) channels in FTL and keep their
    # original form)
    spatialFTLRatio: float = 1.0
    # ratio of the n_channel for temporal FTL (i.e., we won't apply
    # transformation for the rest of (1-ratio) channels in FTL and keep their
    # original form)
    temporalFTLRatio: float = 1.0
    # #wrist_rigid_points used for wrist transformation SVD
    nWristRigidPts: int = 7
