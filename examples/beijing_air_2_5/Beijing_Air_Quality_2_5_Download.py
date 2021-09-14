import requests

def download_file_from_google_drive(id, destination):
    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    print('Start downloading')

    file_links = []

    with open('download_link.txt', 'r') as f:
        for line in f.readlines():
            if 'file' in line:
                start_str = 'file/d/'
                end_str = '/view?usp'
                
                start_idx = line.find(start_str) + len(start_str)
                end_idx = line.find(end_str)
                
                name_idx = line.find(' ')
                
                file_links.append([line[start_idx:end_idx], line[name_idx+1:-1]])

    for f in file_links:
        file_id = f[0]
        destination = f[1]
        download_file_from_google_drive(file_id, destination)
    
    print('Done')
