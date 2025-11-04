import secrets
import uuid
import logging
from typing import Optional

# Third-party libraries
from fastapi import FastAPI, Request, status, HTTPException
from starlette.responses import JSONResponse
from itsdangerous import Signer, BadSignature

#SAME_SITE: "none" | "lax" | "strict" = "none"
SAME_SITE =  "none"
SECRET_KEY = "SUPER_SECRET"
COOKIE_NAME = "user_session"
COOKIE_SECURE = (SAME_SITE == "none") # Must be True when SameSite is "none"

app = FastAPI(title="CSRF Demo") # Use the provided title
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# The signer object handles HMAC signing and verification of the cookie value
signer = Signer(SECRET_KEY, salt=COOKIE_NAME)


# --- Utility Functions ---
def sign_data(data: str) -> bytes:
    """Signs the input string data using the global signer."""
    return signer.sign(data.encode('utf-8'))

def unsign_cookie(signed_cookie: str) -> Optional[str]:
    """Unsigns and verifies the cookie. Returns data if valid, None if invalid."""
    try:
        # unsign() verifies the signature using SECRET_KEY.
        # It raises BadSignature if the key has rotated or the value is tampered.
        unsigned_data = signer.unsign(signed_cookie.encode('utf-8'))
        return unsigned_data.decode('utf-8')
    except BadSignature:
        logging.error(f"ATTENTION: Invalid signature detected for cookie '{COOKIE_NAME}'. "
                      f"Possible expired key or tampering.")
        return None


@app.get("/", status_code=status.HTTP_200_OK)
async def login():
    """
    GET endpoint that sets a signed, persistent session cookie using Starlette's JSONResponse.
    
    This simulates a successful login or session creation.
    """
    
    # 1. Autogenerate a persistent user ID (simulating a permanent user record)
    user_id = str(uuid.uuid4())
    
    # 2. Sign the user ID
    signed_user_id_bytes = sign_data(user_id)
    signed_user_id = signed_user_id_bytes.decode('utf-8')
    
    # 3. Create Starlette JSONResponse object
    content = {"message": "Session created and signed cookie set.", "user_id": user_id}
    response = JSONResponse(content)
    
    # 4. Set the cookie using the response object's set_cookie method
    response.set_cookie(
        key=COOKIE_NAME,
        value=signed_user_id,
        httponly=True,
        secure=COOKIE_SECURE,  # MUST be True if SameSite is "None"
        samesite=SAME_SITE, # Starlette requires the string literal here
        max_age=3600 * 24 * 30  # Persistent for 30 days
    )
    
    logging.info(f"SET COOKIE: Successfully set session cookie for User ID: {user_id}")
    
    return response

@app.post("/action", status_code=status.HTTP_200_OK)
async def action(request: Request):
    """
    POST endpoint that REQUIRES the signed cookie to authenticate the action.
    
    FastAPI's Request object already uses Starlette's underlying cookie retrieval.
    """
    # 1. Retrieve the raw signed cookie value
    signed_cookie = request.cookies.get(COOKIE_NAME)
    
    if not signed_cookie:
        logging.warning("REJECTED: POST request blocked. No session cookie found.")
        raise HTTPException(status_code=401, detail="Unauthorized: No session cookie.")
    
    # 2. Unsign and verify the cookie
    user_id = unsign_cookie(signed_cookie)
    
    if not user_id:
        # BadSignature was caught in unsign_cookie()
        logging.warning(f"REJECTED: POST request blocked. Invalid or tampered cookie received: {signed_cookie[:20]}...")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid session signature.")
    
    # 3. Action executed (only if cookie is valid)
    logging.info(f"EXECUTED: Executing POST for user {user_id}")
    
    return {"message": f"Action successfully executed for user {user_id}"}

@app.get("/")
async def welcome():
    """Simple health check endpoint."""
    return {"status": "Server running", "cookie_name": COOKIE_NAME, "same_site": SAME_SITE}

if __name__ == "__main__":
    import uvicorn
    print(f"Running server with SECRET_KEY: {SECRET_KEY}")
    print(f"SameSite setting: {SAME_SITE}. Remember this requires HTTPS/Secure.")
    print("running http://localhost:3537")
    uvicorn.run(app, host="0.0.0.0", port=3537)

