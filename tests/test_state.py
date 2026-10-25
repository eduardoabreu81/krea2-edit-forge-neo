"""Which pictures and reference sets a generation uses, and when the cached conditioning has to go."""

import unittest

from forge_krea2edit import state
from forge_krea2edit.state import Options, ReferenceSet


class StateTest(unittest.TestCase):
    def setUp(self):
        state.reset()

    def tearDown(self):
        state.reset()

    def test_changing_options_clears_the_cache(self):
        self.assertTrue(state.start(Options()))
        self.assertFalse(state.start(Options()))
        self.assertTrue(state.start(Options(subject_boost=2.0)))
        self.assertTrue(state.start(None))
        self.assertFalse(state.start(None))

    def test_later_conditionings_of_a_generation_keep_the_pictures(self):
        state.start(Options())
        scene, subject = object(), object()
        self.assertEqual(state.references_for([scene, subject]), [scene, subject])
        # the hires pass: Forge has handed img2img's picture over already
        self.assertEqual(state.references_for([subject]), [scene, subject])
        state.start(Options())
        self.assertEqual(state.references_for([subject]), [subject])

    def test_lookup_by_size(self):
        forge_list = ["latent"]
        first, hires = ReferenceSet(["a"], True), ReferenceSet(["b"], True)
        state.store((64, 64), first, forge_list)
        state.store((128, 128), hires, forge_list)
        self.assertEqual(state.lookup((64, 64), forge_list), (first, True))
        self.assertEqual(state.lookup((128, 128), forge_list), (hires, True))
        self.assertEqual(state.lookup((96, 96), forge_list), (first, False))

    def test_forge_emptying_its_list_retires_the_sets(self):
        forge_list = ["latent"]
        state.store((64, 64), ReferenceSet(["a"], True), forge_list)
        forge_list.clear()  # dynamic_args.reset() at a model load
        self.assertEqual(state.lookup((64, 64), forge_list), (None, False))
        self.assertEqual(state.lookup((64, 64), ["other"]), (None, False))


if __name__ == "__main__":
    unittest.main()
