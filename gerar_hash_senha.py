import getpass
import hashlib

senha = getpass.getpass("Digite a senha que você quer usar no painel de treino: ")
hash_senha = hashlib.sha256(senha.encode("utf-8")).hexdigest()

print("\nColoque isso na variável de ambiente HATEBR_SENHA_HASH (não a senha em si):\n")
print(hash_senha)
print("\nPowerShell: setx HATEBR_SENHA_HASH \"" + hash_senha + "\"")
