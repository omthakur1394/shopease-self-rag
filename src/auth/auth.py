import os 
from dotenv import load_dotenv
from typing import Annotated
from fastapi import Depends,HTTPException,status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError,jwt
from pydantic import BaseModel
load_dotenv()
SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
class TokenData(BaseModel):
    id : int 
    username : str
    email : str

def get_Current_User(token:Annotated[str,Depends(oauth2_scheme)]):
   credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
   try:
       playload = jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
       user_id : int = playload.get("id")
       username : str = playload.get("username")
       email:str = playload.get("email")

       if username is None or user_id is None:
           raise credentials_exception
       return TokenData(id = user_id,username=username,email=email)
   except:
       raise credentials_exception 
   
    