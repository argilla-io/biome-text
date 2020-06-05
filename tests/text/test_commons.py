import pytest

from biome.text.commons import ImmutableDict


class TestImmutableDict:
    def test_cannot_mutate(self):
        dict = ImmutableDict(a=1, b="2", c=1000.00)

        with pytest.raises(TypeError):
            dict.f = "F"

        with pytest.raises(TypeError):
            dict.a = 100

    # TODO: Test a serialization/deserialization
