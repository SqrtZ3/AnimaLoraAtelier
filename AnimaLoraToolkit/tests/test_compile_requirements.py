import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.objective import validate_compile_requirements


class TestValidateCompileRequirements(unittest.TestCase):

    def test_compile_off_always_passes(self):
        """compile=False is a no-op regardless of other flags."""
        result = validate_compile_requirements(False, False, False)
        self.assertIsNone(result)

    def test_compile_on_all_prereqs_passes(self):
        """compile=True with both prerequisites satisfied must not raise."""
        result = validate_compile_requirements(True, True, True)
        self.assertIsNone(result)

    def test_compile_on_missing_fit_packed_raises(self):
        """compile=True without fit_packed_training must raise RuntimeError
        mentioning fit_packed_training."""
        with self.assertRaisesRegex(RuntimeError, "fit_packed_training"):
            validate_compile_requirements(True, False, True)

    def test_compile_on_missing_token_bucket_raises(self):
        """compile=True without token_bucket must raise RuntimeError
        mentioning token_bucket."""
        with self.assertRaisesRegex(RuntimeError, "token_bucket"):
            validate_compile_requirements(True, True, False)

    def test_compile_on_module_dropout_now_passes(self):
        """module_dropout>0 is now compile-safe (keep scalar pre-drawn outside the
        compiled region), so it must NOT raise even with compile on."""
        self.assertIsNone(validate_compile_requirements(True, True, True, 0.1))

    def test_compile_on_zero_module_dropout_passes(self):
        """module_dropout=0 with all prereqs satisfied must not raise."""
        self.assertIsNone(validate_compile_requirements(True, True, True, 0.0))


if __name__ == "__main__":
    unittest.main()
