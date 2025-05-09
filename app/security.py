

import os
import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional


from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


from dotenv import load_dotenv

logger = logging.getLogger("app.security")


from jose import JWTError, jwt
from passlib.context import CryptContext


from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer




load_dotenv()



try:
    _master_key_b64 = os.getenv('RINI_API_MASTER_KEY')
    if not _master_key_b64:
        logger.error("환경 변수 'RINI_API_MASTER_KEY'가 설정되지 않았습니다.")
        raise ValueError("환경 변수 'RINI_API_MASTER_KEY'가 설정되지 않았습니다.")
    MASTER_KEY = base64.urlsafe_b64decode(_master_key_b64)
    if len(MASTER_KEY) != 32:
        logger.error(f"마스터 키 길이가 유효하지 않습니다. (길이: {len(MASTER_KEY)}, 필요: 32)")
        raise ValueError("마스터 키는 반드시 32바이트여야 합니다.")
    logger.info("RINI_API_MASTER_KEY 로드 및 검증 완료.")
except Exception as e:
    logger.critical(f"환경 변수 로드 또는 마스터 키 처리 중 심각한 오류 발생: {e}", exc_info=True)
    MASTER_KEY = None

def encrypt_api_key(api_key: str) -> str:
    """주어진 API 키(문자열)를 AES-GCM 방식으로 암호화하고,
       nonce와 암호문을 base64로 인코딩하여 결합한 문자열을 반환합니다."""
    if MASTER_KEY is None:
        logger.error("암호화 시도 실패: 마스터 키가 초기화되지 않았습니다.")
        raise RuntimeError("암호화 마스터 키가 초기화되지 않았습니다.")
    if not isinstance(api_key, str) or not api_key:
        logger.warning("잘못된 API 키 입력: API 키는 비어 있지 않은 문자열이어야 합니다.")
        raise ValueError("API 키는 비어 있지 않은 문자열이어야 합니다.")

    try:
        logger.debug("API 키 암호화 시작.")
        aesgcm = AESGCM(MASTER_KEY)
        nonce = os.urandom(12)
        api_key_bytes = api_key.encode('utf-8')
        encrypted_data = aesgcm.encrypt(nonce, api_key_bytes, None)


        encrypted_payload = base64.urlsafe_b64encode(nonce) + b'.' + base64.urlsafe_b64encode(encrypted_data)
        logger.debug("API 키 암호화 완료.")
        return encrypted_payload.decode('utf-8')
    except Exception as e:
        logger.exception(f"API 키 암호화 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"API 키 암호화 중 오류 발생: {e}")


def decrypt_api_key(encrypted_payload: str) -> str:
    """암호화된 페이로드(문자열)를 받아 복호화하고, 원본 API 키(문자열)를 반환합니다."""
    if MASTER_KEY is None:
        logger.error("복호화 시도 실패: 마스터 키가 초기화되지 않았습니다.")
        raise RuntimeError("암호화 마스터 키가 초기화되지 않았습니다.")
    if not isinstance(encrypted_payload, str) or '.' not in encrypted_payload:
        logger.warning("잘못된 형식의 암호화된 페이로드 입력.")
        raise ValueError("잘못된 형식의 암호화된 페이로드입니다.")

    try:
        logger.debug("API 키 복호화 시작.")
        nonce_b64, encrypted_data_b64 = encrypted_payload.encode('utf-8').split(b'.', 1)
        nonce = base64.urlsafe_b64decode(nonce_b64)
        encrypted_data = base64.urlsafe_b64decode(encrypted_data_b64)

        aesgcm = AESGCM(MASTER_KEY)
        decrypted_data = aesgcm.decrypt(nonce, encrypted_data, None)
        logger.debug("API 키 복호화 완료.")
        return decrypted_data.decode('utf-8')
    except InvalidTag as e:
        logger.error(f"API 키 복호화 실패 (InvalidTag): {e}. 페이로드 또는 키가 손상되었거나 잘못되었을 수 있습니다.")
        raise ValueError(f"API 키 복호화 실패: 인증 태그가 유효하지 않습니다.")
    except (ValueError, IndexError) as e:
        logger.error(f"API 키 복호화 실패 (ValueError/IndexError): {e}. 페이로드 형식이 잘못되었습니다.")
        raise ValueError(f"API 키 복호화 실패: 페이로드 형식이 잘못되었습니다.")
    except Exception as e:
        logger.exception(f"API 키 복호화 중 예기치 않은 오류 발생: {e}")
        raise RuntimeError(f"API 키 복호화 중 예기치 않은 오류 발생: {e}")




pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """입력된 평문 비밀번호와 해시된 비밀번호가 일치하는지 검증합니다."""
    logger.debug("비밀번호 검증 시도.")
    verified = pwd_context.verify(plain_password, hashed_password)
    logger.debug(f"비밀번호 검증 결과: {'성공' if verified else '실패'}")
    return verified

def get_password_hash(password: str) -> str:
    """평문 비밀번호를 받아 해시된 문자열을 반환합니다."""
    logger.debug("비밀번호 해싱 시도.")
    hashed_password = pwd_context.hash(password)
    logger.debug("비밀번호 해싱 완료.")
    return hashed_password



JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))

if not JWT_SECRET_KEY:
    logger.warning("환경 변수 'JWT_SECRET_KEY'가 설정되지 않았습니다. 임시 키를 사용합니다. (보안 위험!)")
    JWT_SECRET_KEY = "a_very_insecure_temporary_secret_key_for_dev_only"
else:
    logger.info("JWT_SECRET_KEY 로드 완료.")

logger.info(f"JWT 알고리즘: {JWT_ALGORITHM}, 토큰 만료 시간: {ACCESS_TOKEN_EXPIRE_MINUTES}분")

bearer_scheme = HTTPBearer(scheme_name="JWT Bearer Token", description="Enter JWT Bearer token **_only_**")
logger.debug("HTTPBearer scheme 초기화 완료.")







class TokenData(BaseModel):
    """JWT 토큰의 페이로드(payload) 데이터 형태 정의 (주로 subject를 포함)"""
    sub: Optional[str] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """주어진 데이터(payload)와 만료 시간을 사용하여 JWT 액세스 토큰을 생성합니다."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    logger.debug(f"액세스 토큰 생성을 위한 페이로드: {to_encode}")

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    logger.info(f"액세스 토큰 생성 완료 (sub: {data.get('sub')}, exp: {expire.isoformat()}).")
    return encoded_jwt

async def verify_token_and_get_sub(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    """HTTP Bearer 토큰을 검증하고 'sub' 클레임(사용자 식별자)을 추출합니다."""
    logger.debug("토큰 검증 및 'sub' 클레임 추출 시도.")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = credentials.credentials
    logger.debug(f"수신된 토큰 (일부): {token[:20]}..." if token else "토큰 없음")
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        logger.debug(f"토큰 페이로드 디코딩 성공: {payload}")
        subject: Optional[str] = payload.get("sub")
        if subject is None:
            logger.warning("토큰에 'sub' 클레임이 없습니다.")
            raise credentials_exception
        logger.info(f"'sub' 클레임 추출 성공: {subject}")
    except JWTError as e:
        logger.error(f"JWT 오류 발생: {e}", exc_info=True)
        raise credentials_exception
    except Exception as e:
        logger.exception(f"토큰 검증 중 예기치 않은 오류 발생: {e}")
        raise credentials_exception
    return subject


async def get_current_user_id(subject: str = Depends(verify_token_and_get_sub)) -> int:
    """토큰에서 추출한 subject(사용자 식별자)를 사용하여 사용자 ID(int)를 반환합니다."""
    logger.debug(f"현재 사용자 ID 변환 시도 (subject: {subject}).")
    try:
        user_id = int(subject)
        logger.info(f"사용자 ID 변환 성공: {user_id}")      
    except ValueError:
        logger.error(f"토큰의 'sub' 클레임({subject})을 정수형 사용자 ID로 변환할 수 없습니다.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier in token",
        )
    return user_id

"""
async def verify_token_and_get_sub(token: str = Depends(oauth2_scheme)) -> str:
    "#""토큰을 검증하고 'sub' 클레임(사용자 식별자)을 추출하는 의존성 함수."#""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        subject: Optional[str] = payload.get("sub")
        if subject is None:
            raise credentials_exception
        # 여기서 TokenData 스키마로 유효성 검사를 추가할 수도 있습니다.
        # token_data = TokenData(sub=subject)
    except JWTError: # jose.exceptions.JWTError
        raise credentials_exception
    return subject

# --- 현재 사용자 정보 가져오기 의존성 함수 ---
# schemas.py 에서 User 모델을 임포트해야 합니다.
from .schemas import User as UserSchema

# 이 함수는 API 엔드포인트에서 Depends를 통해 사용되어,
# 토큰을 검증하고 해당 사용자의 ID를 반환합니다.
async def get_current_user_id(subject: str = Depends(verify_token_and_get_sub)) -> int:
    "#""토큰에서 추출한 subject(사용자 식별자)를 사용하여 사용자 ID(int)를 반환합니다."#""
    try:
        # JWT의 'sub' 클레임에 사용자 ID(숫자)가 문자열로 저장되어 있다고 가정합니다.
        user_id = int(subject)
        # 여기서 user_id를 사용하여 DB에서 실제 사용자 존재 여부를 확인할 수도 있습니다 (선택 사항).
        # user = crud.get_user(db, user_id=user_id)
        # if user is None:
        #     raise HTTPException(status_code=404, detail="User not found")
    except ValueError:
        # 'sub' 클레임이 숫자로 변환될 수 없는 경우 (예: 이메일이 저장된 경우)
        # 또는 다른 방식의 사용자 식별이 필요한 경우 이 부분을 수정해야 합니다.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier in token",
        )
    return user_id
"""










