# Notes Service

A tiny notes management service used as a demo target for CodeWeaver.

It intentionally contains a handful of realistic engineering issues:

- a failing test in `tests/test_api.py` (sort-order regression)
- duplicated normalization logic between `app/api.py` and `app/storage.py`
- a hardcoded API token in `app/config.py`
- a committed `.env` file
- a bare `except` swallowing errors in `app/storage.py`
- list endpoints without pagination

## Layout

```
app/          application code
tests/        pytest suite
tools/        maintenance scripts
```

## Running tests

```
python -m pytest -q
```
