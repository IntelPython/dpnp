
:: CALL "1.env.bat"

python setup.py clean
python setup.py build_clib_setuptools

:: inplace build
python setup.py build_ext --inplace

:: development build. Root privileges needed
:: python setup.py develop
