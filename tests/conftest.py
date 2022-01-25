from pytest import fixture

from data.sin import sin_data as sindata, SinData

@fixture
def sin_data() -> SinData:
    return sindata()
