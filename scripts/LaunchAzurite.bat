rem チェック対象のディレクトリを指定
SET dir=%CD%\azurite

rem ディレクトリが存在するかチェックする
If not exist %dir% mkdir %dir%

docker run -d --rm -p 10000:10000 -p 10001:10001 -v %dir%:/data mcr.microsoft.com/azure-storage/azurite
