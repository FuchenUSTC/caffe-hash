set ROOTDIR=../../

set EXAMPLE=%ROOTDIR%examples/mnist
set DATA=%ROOTDIR%data/mnist
set BUILD=%ROOTDIR%bin

REM set BACKEND=lmdb
set BACKEND=leveldb

echo "Creating %BACKEND%..."

rd /s /q "%EXAMPLE%/mnist_train_%BACKEND%"
rd /s /q "%EXAMPLE%/mnist_test_%BACKEND%"

"%BUILD%/convert_mnist_data.exe" %DATA%/train-images-idx3-ubyte %DATA%/train-labels-idx1-ubyte %EXAMPLE%/mnist_train_%BACKEND% --backend=%BACKEND%
"%BUILD%/convert_mnist_data.exe" %DATA%/t10k-images-idx3-ubyte %DATA%/t10k-labels-idx1-ubyte %EXAMPLE%/mnist_test_%BACKEND% --backend=%BACKEND%

echo "Done."

pause