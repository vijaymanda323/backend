import instaloader

L = instaloader.Instaloader()

# Login with your username and password
username = 'aparna_chowdhary_2244'
password = 'Aparna@0987'

L.login(username, password)  # May raise an exception if credentials are invalid

# Save session to avoid repeated login
L.save_session_to_file()
