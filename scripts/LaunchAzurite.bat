rem �`�F�b�N�Ώۂ̃f�B���N�g�����w��
SET dir=%CD%\azurite

rem �f�B���N�g�������݂��邩�`�F�b�N����
If not exist %dir% mkdir %dir%

docker run -d --rm -p 10000:10000 -p 10001:10001 -v %dir%:/data mcr.microsoft.com/azure-storage/azurite
