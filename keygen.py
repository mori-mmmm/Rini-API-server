from cryptography.fernet import Fernet


import os
master_key = os.urandom(32)
import base64
encoded_master_key = base64.urlsafe_b64encode(master_key)
print(f"생성된 마스터 키 (base64 인코딩됨, 안전하게 보관하세요):\n{encoded_master_key.decode()}")
