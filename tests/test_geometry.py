"""Fitting, positions and the boost bias against upstream's (comfyui-krea2edit 2.0.0)."""

import unittest

import torch

from forge_krea2edit import geometry
from tests.reference import krea2edit_comfy


class IdentityVAE:
    def encode(self, pixels):
        return pixels


def picture(h, w, seed=0):
    return torch.rand(1, h, w, 3, generator=torch.Generator().manual_seed(seed))


class FitTest(unittest.TestCase):
    # picture sizes against result latent sizes: same proportions, near-matched, portrait in landscape, landscape in
    # portrait, odd sizes, a picture smaller than the result
    CASES = [((1024, 1024), (128, 128)), ((1000, 1024), (128, 128)), ((1536, 1024), (96, 160)),
             ((768, 1360), (160, 112)), ((753, 1129), (128, 96)), ((300, 200), (128, 128)), ((577, 901), (64, 64))]

    def test_matches_upstream_pixel_path(self):
        for (ih, iw), (lat_h, lat_w) in self.CASES:
            image = picture(ih, iw)
            for ours, theirs in ((geometry.FIT, "fit"), (geometry.CROP, "crop")):
                with self.subTest(picture=(ih, iw), latent=(lat_h, lat_w), mode=ours):
                    expected = krea2edit_comfy._fit_encode_image(image, IdentityVAE(), lat_h, lat_w, {}, ("a",), theirs)
                    torch.testing.assert_close(geometry.fit_pixels(image, lat_h, lat_w, ours), expected)

    def test_fit_lands_on_the_token_grid_inside_the_result(self):
        for (ih, iw), (lat_h, lat_w) in self.CASES:
            _, (nh, nw) = geometry.fit_plan(ih, iw, lat_h, lat_w, geometry.FIT)
            self.assertEqual((nh % 16, nw % 16), (0, 0))
            self.assertLessEqual(nh, lat_h * 8)
            self.assertLessEqual(nw, lat_w * 8)

    def test_crop_is_the_result_size(self):
        for (ih, iw), (lat_h, lat_w) in self.CASES:
            self.assertEqual(geometry.fit_plan(ih, iw, lat_h, lat_w, geometry.CROP)[1], (lat_h * 8, lat_w * 8))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            geometry.fit_plan(64, 64, 8, 8, "stretch")


class PositionTest(unittest.TestCase):
    def test_reference_ids_match_upstream(self):
        grids = [(6, 8), (6, 4), (3, 8)]
        for centered in (False, True):
            ours = geometry.reference_ids(2, grids, 6, 8, centered, "cpu")
            if centered:
                theirs = [krea2edit_comfy._imgids_offset(2, i + 1, gh, gw, 6, 8, "cpu") for i, (gh, gw) in enumerate(grids)]
            else:
                theirs = [krea2edit_comfy._imgids(2, i + 1, gh, gw, "cpu") for i, (gh, gw) in enumerate(grids)]
            for a, b in zip(ours, theirs):
                torch.testing.assert_close(a, b)

    def test_place_matches_upstream(self):
        names = {geometry.ALIGNED: "aligned (trained)", geometry.BESIDE: "beside (right of target)",
                 geometry.BAG: "bag of features"}
        ids = geometry.reference_ids(1, [(6, 8), (6, 4), (6, 8)], 6, 8, True, "cpu")
        for subject in geometry.POSITIONS:
            for scene in geometry.POSITIONS:
                with self.subTest(subject=subject, scene=scene):
                    modes = geometry.placements(3, subject, scene)
                    theirs = krea2edit_comfy._place_refs(ids, [names[m] for m in modes], 6, 8)
                    for a, b in zip(geometry.place(ids, modes, 6, 8), theirs):
                        torch.testing.assert_close(a, b)

    def test_placements_name_the_last_picture_subject(self):
        self.assertEqual(geometry.placements(1, "s", "a"), ["s"])
        self.assertEqual(geometry.placements(3, "s", "a"), ["a", "a", "s"])

    def test_beside_leaves_no_overlap(self):
        ids = geometry.reference_ids(1, [(6, 8), (6, 8)], 6, 8, False, "cpu")
        placed = geometry.place(ids, [geometry.BESIDE, geometry.BESIDE], 6, 8)
        self.assertEqual(placed[0][..., 2].min().item(), 8)
        self.assertEqual(placed[1][..., 2].min().item(), 16)


class BiasTest(unittest.TestCase):
    def test_matches_upstream(self):
        for boosts in ([2.0], [1.0, 3.0], [0.5, 1.0], [4.0, 2.0, 0.0]):
            lens = [12, 8, 10][: len(boosts)]
            with self.subTest(boosts=boosts):
                ours = geometry.attention_bias(boosts, 5, lens, 9, "cpu", torch.float32)
                theirs = krea2edit_comfy._ref_attn_bias(boosts, None, 5, lens, 9, None, "cpu", torch.float32)
                torch.testing.assert_close(ours, theirs)

    def test_no_boost_no_bias(self):
        self.assertIsNone(geometry.attention_bias([1.0, 1.0], 5, [4, 4], 4, "cpu", torch.float32))


class GroundingTest(unittest.TestCase):
    def test_longest_side(self):
        self.assertEqual(geometry.grounding_size(1024, 1536, 768), (512, 768))
        self.assertEqual(geometry.grounding_size(600, 400, 768), (600, 400))
        self.assertEqual(geometry.grounding_size(2048, 1024, 0), (2048, 1024))


if __name__ == "__main__":
    unittest.main()
