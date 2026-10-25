"""The V2 forward against upstream's (comfyui-krea2edit 2.0.0) and Forge Neo's own, on a tiny stand-in DiT whose
blocks read the positions and the attention bias, so placement and boost show in the output."""

import unittest

import torch
import torch.nn as nn

from tests import forge_stubs
from tests.reference import forge_krea, krea2edit_comfy


class Fusion(nn.Module):
    def forward(self, context, mask=None, transformer_options={}):
        return context.mean(dim=2)


class Block(nn.Module):
    """Attention over the whole sequence with a distance penalty from the positions and the additive bias."""

    def __init__(self, dim):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        scores = self.q(x) @ self.k(x).transpose(-1, -2) / x.shape[-1] ** 0.5 - torch.cdist(freqs, freqs)
        if mask is not None:
            scores = scores + mask[:, 0]
        return x + torch.softmax(scores, dim=-1) @ self.v(x) + vec.mean(dim=-1, keepdim=True)


class Last(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.linear = nn.Linear(dim, out)

    def forward(self, x, t):
        return self.linear(x) + t.mean(dim=-1, keepdim=True)


class TinyDiT(nn.Module):
    patch, channels, tdim, dim, txtdim = 2, 4, 16, 24, 12

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first = nn.Linear(self.channels * self.patch**2, self.dim)
        self.tmlp = nn.Linear(self.tdim, self.dim)
        self.tproj = nn.Linear(self.dim, self.dim)
        self.txtfusion = Fusion()
        self.txtmlp = nn.Linear(self.txtdim, self.dim)
        self.blocks = nn.ModuleList([Block(self.dim) for _ in range(2)])
        self.last = Last(self.dim, self.channels * self.patch**2)

    @staticmethod
    def pe_embedder(pos):
        return pos

    @staticmethod
    def _unpack_context(context):
        return context


def latent(bs, h, w, seed):
    return torch.randn(bs, TinyDiT.channels, h, w, generator=torch.Generator().manual_seed(seed))


class ForwardTest(unittest.TestCase):
    def setUp(self):
        self.stubs = forge_stubs.installed(forge_stubs.backend())
        self.stubs.__enter__()
        from forge_krea2edit import forward, geometry, state

        self.forward, self.geometry, self.state = forward, geometry, state
        self.m = TinyDiT().eval()
        self.x = latent(1, 12, 16, 1).unsqueeze(2)
        self.t = torch.tensor([0.7])
        self.context = torch.randn(1, 5, 3, TinyDiT.txtdim, generator=torch.Generator().manual_seed(2))

    def tearDown(self):
        self.stubs.__exit__(None, None, None)

    def ours(self, refs, options, centered=False, exact=True, x=None):
        x = self.x if x is None else x
        with torch.no_grad():
            return self.forward.edit_forward(self.m, x, self.t, self.context, {}, self.state.ReferenceSet(refs, centered),
                                             exact, options)

    def upstream(self, refs, x=None, **kwargs):
        x = self.x if x is None else x
        with torch.no_grad():
            return krea2edit_comfy.krea2_edit_forward(self.m, x, self.t, self.context, refs, {}, **kwargs)

    def forge(self, refs, x=None):
        x = self.x if x is None else x
        with torch.no_grad():
            return forge_krea.forward(self.m, x, self.t, self.context, refs)

    def test_neutral_options_are_forge_own_edit(self):
        refs = [latent(1, 12, 16, 3)]
        torch.testing.assert_close(self.ours(refs, self.state.Options()), self.forge(refs))

    def test_neutral_options_resize_as_forge_does(self):
        refs = [latent(1, 20, 14, 3), latent(1, 12, 16, 4)]
        torch.testing.assert_close(self.ours(refs, self.state.Options(), exact=False), self.forge(refs))

    def test_cond_uncond_batch_repeats_the_references(self):
        x = latent(2, 12, 16, 5).unsqueeze(2)
        self.context = self.context.repeat(2, 1, 1, 1)
        refs = [latent(1, 12, 16, 3)]
        torch.testing.assert_close(self.ours(refs, self.state.Options(), x=x), self.forge(refs, x=x))

    def test_single_reference_boost_and_beside(self):
        refs = [latent(1, 12, 16, 3)]
        options = self.state.Options(subject_boost=2.0, subject_position=self.geometry.BESIDE)
        expected = self.upstream(refs, ref_boost=2.0, ref_position="beside (right of target)")
        torch.testing.assert_close(self.ours(refs, options), expected)
        self.assertFalse(torch.allclose(expected, self.forge(refs)))

    def test_scene_aligned_subject_beside(self):
        refs = [latent(1, 12, 16, 3), latent(1, 12, 16, 4)]
        options = self.state.Options(subject_boost=3.0, subject_position=self.geometry.BESIDE, scene_boost=0.5)
        expected = self.upstream(refs, ref_boost=3.0, ref_boost_a=0.5, ref_position="beside (right of target)")
        torch.testing.assert_close(self.ours(refs, options), expected)

    def test_both_beside_and_bag(self):
        refs = [latent(1, 12, 16, 3), latent(1, 12, 16, 4)]
        options = self.state.Options(subject_position=self.geometry.BAG, scene_position=self.geometry.BESIDE)
        expected = self.upstream(refs, ref_position="bag of features", ref_position_a="beside (right of target)")
        torch.testing.assert_close(self.ours(refs, options), expected)

    def test_fit_keeps_a_smaller_reference_centered(self):
        # a 12 x 8 latent (portrait picture) fitted inside a 12 x 16 result
        refs = [latent(1, 12, 8, 3)]
        options = self.state.Options(subject_boost=2.0)
        expected = self.upstream(refs, ref_boost=2.0, ref_native=True, pos_mode="stride1")
        torch.testing.assert_close(self.ours(refs, options, centered=True), expected)

    def test_fit_beside_with_a_scene(self):
        refs = [latent(1, 12, 16, 3), latent(1, 8, 12, 4)]
        options = self.state.Options(subject_boost=2.5, subject_position=self.geometry.BESIDE)
        expected = self.upstream(refs, ref_boost=2.5, ref_native=True, pos_mode="stride1",
                                 ref_position="beside (right of target)")
        torch.testing.assert_close(self.ours(refs, options, centered=True), expected)

    def test_fit_for_another_size_falls_back_to_forge_resize(self):
        refs = [latent(1, 12, 8, 3)]
        torch.testing.assert_close(self.ours(refs, self.state.Options(), centered=True, exact=False), self.forge(refs))


if __name__ == "__main__":
    unittest.main()
