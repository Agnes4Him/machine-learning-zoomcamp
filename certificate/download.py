from hashlib import sha1

def compute_hash(email):
    return sha1(email.encode('utf-8')).hexdigest()

def compute_certificate_id(email):
    email_clean = email.lower().strip()
    return compute_hash(email_clean + '_')

cohort = 2025
course = 'ml-zoomcamp'
your_id = compute_certificate_id('')
url = f"https://certificate.datatalks.club/{course}/{cohort}/{your_id}.pdf"
print(url)