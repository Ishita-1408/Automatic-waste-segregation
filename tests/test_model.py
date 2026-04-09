"""
Tests for Waste Segregation model utilities.
Run: pytest tests/ -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelArchitecture(unittest.TestCase):
    def test_model_output_shape(self):
        """Model should produce logits of shape (batch, num_classes)."""
        import torch
        from train import build_model
        model = build_model(num_classes=6, freeze_backbone=True)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (2, 6))

    def test_model_num_classes_configurable(self):
        """build_model should respect num_classes parameter."""
        import torch
        from train import build_model
        for n in [3, 6, 10]:
            m = build_model(num_classes=n)
            out = m(torch.randn(1, 3, 224, 224))
            self.assertEqual(out.shape[1], n)


class TestTransforms(unittest.TestCase):
    def test_train_transforms_output_tensor(self):
        from PIL import Image
        from train import train_transforms
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = train_transforms(img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertAlmostEqual(tensor.mean().item(), 0.0, delta=1.5)  # normalized

    def test_val_transforms_deterministic(self):
        from PIL import Image
        from train import val_transforms
        img = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
        t1 = val_transforms(img)
        t2 = val_transforms(img)
        self.assertTrue((t1 == t2).all())


class TestPredictUtils(unittest.TestCase):
    def test_tips_coverage(self):
        """Every class should have a tip entry."""
        from predict import TIPS
        expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
        self.assertEqual(set(TIPS.keys()), expected)

    def test_tips_schema(self):
        from predict import TIPS
        for label, info in TIPS.items():
            self.assertIn("bin", info, f"Missing 'bin' for {label}")
            self.assertIn("tip", info, f"Missing 'tip' for {label}")
            self.assertIn("recyclable", info, f"Missing 'recyclable' for {label}")
            self.assertIsInstance(info["recyclable"], bool)

    def test_trash_not_recyclable(self):
        from predict import TIPS
        self.assertFalse(TIPS["trash"]["recyclable"])

    def test_all_others_recyclable(self):
        from predict import TIPS
        for cls in ["cardboard", "glass", "metal", "paper", "plastic"]:
            self.assertTrue(TIPS[cls]["recyclable"])


class TestTrainingHistory(unittest.TestCase):
    def test_plot_history_creates_file(self):
        from train import plot_history
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "train_loss": [0.8, 0.5, 0.3],
                "val_loss": [0.9, 0.6, 0.4],
                "train_acc": [0.6, 0.75, 0.88],
                "val_acc": [0.55, 0.70, 0.85],
            }
            plot_history(history, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "training_history.png")))


class TestConfusionMatrix(unittest.TestCase):
    def test_confusion_matrix_creates_file(self):
        from train import plot_confusion_matrix
        labels = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        preds = [0, 1, 2, 3, 4, 5, 1, 1, 2, 3]
        names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_confusion_matrix(labels, preds, names, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "confusion_matrix.png")))


if __name__ == "__main__":
    unittest.main()
