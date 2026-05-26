"""
Descriptor Heads for BEV Place Recognition
"""

from .latent_descriptor import LatentDescriptorHead
from .netvlad_descriptor import NetVLADDescriptor, create_netvlad_descriptor

__all__ = ['LatentDescriptorHead', 'NetVLADDescriptor', 'create_netvlad_descriptor']
