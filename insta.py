import instaloader

# Replace with your actual Instagram credentials
username = "aparna_chowdhary_2244"
password = "Aparna@0987"  # Add your Instagram password

# Initialize Instaloader instance
loader = instaloader.Instaloader()

try:
    loader.load_session_from_file(username)
except FileNotFoundError:
    print("No session found, logging in...")

# Login using username and password
loader.context.username = username
loader.context.password = password
loader.context.login(username, password)

# Save session to file
session_filename = f"session-{username}"  # This will create 'session-_vijay.manda'
loader.save_session_to_file(session_filename)

print(f"Session file saved as {session_filename}")
