import secrets
from auth_models import Client
from database import SessionLocal


session = SessionLocal()

new_key = secrets.token_hex(32)
name=input("Enter clients name :")

new_client = Client(
    name =name,
    api_key = new_key

)

session.add(new_client)
session.commit()
session.close()

print(f"Client {name} created with API Key : {new_key}")