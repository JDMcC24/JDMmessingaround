import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = 'http://ufcstats.com/statistics/fighters'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Function to scrape the main fighter statistics page
def scrape_fighter_list(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    fighters = []

    # Find all fighter links
    table = soup.find('table', class_='b-statistics__table')
    links = table.find_all('a', class_='b-link b-link_style_black')
    
    for link in links:
        fighters.append({
            'name': link.text.strip(),
            'url': link['href']
        })
        
    return fighters

# Function to scrape individual fighter pages
def scrape_fighter_data(fighter):
    response = requests.get(fighter['url'], headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    data = {
        'name': fighter['name'],
        'url': fighter['url']
    }
    
    # Extract detailed statistics from the fighter's page
    bio_data = soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
    for item in bio_data:
        stat_name = item.find('i').text.strip() if item.find('i') else None
        stat_value = item.find('span', class_='b-list__box-list-item-number').text.strip() if item.find('span', class_='b-list__box-list-item-number') else None

        if stat_name and stat_value:
            data[stat_name] = stat_value

    return data

# Scrape the list of fighters
fighters = scrape_fighter_list(base_url)

# Scrape detailed data for each fighter
fighter_data = []
for fighter in fighters:
    try:
        fighter_data.append(scrape_fighter_data(fighter))
        time.sleep(1)  # Be respectful and don't overload the server
    except Exception as e:
        print(f"Error scraping data for {fighter['name']}: {e}")

# Save the data to a CSV file
df = pd.DataFrame(fighter_data)
df.to_csv('ufc_fighter_data.csv', index=False)

print('Data scraped and saved to ufc_fighter_data.csv')
