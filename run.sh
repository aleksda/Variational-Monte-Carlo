### Python
# Procedures for compiling and running executables that reproduces results
# and tests the implementation
pytest -ra
python3 src/main.py

### C++
# Procedures for compiling and running executables that reproduces results
# and tests/benchmarks the implementation
make test.x
make main.x
./main.x
make clean
