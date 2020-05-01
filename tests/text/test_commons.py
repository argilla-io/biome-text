import pytest

from biome.text.api_new.commons import InmutableDict


class TestInmutableDict:
    def test_cannot_mutate(self):
        dict = InmutableDict(a=1, b="2", c=1000.00)

        with pytest.raises(TypeError):
            dict.f = "F"

        with pytest.raises(TypeError):
            dict.a = 100
