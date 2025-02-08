import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

start = time.time()


# Base URL for fighter statistics (A-Z pages)
BASE_URL = "http://ufcstats.com/statistics/fighters?char={}&page=all"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# List to store fighter data
fighters_data = []

# Function to scrape individual fighter stats
def scrape_fighter_stats(fighter_url):
    response = requests.get(fighter_url)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    stats = {
        "Name": soup.find("span", class_="b-content__title-highlight").text.strip() if soup.find("span", class_="b-content__title-highlight") else "",
        "Height": "", "Weight": "", "Reach": "", "Stance": "", "DOB": "",
        "SLpM": "", "Str. Acc.": "", "SApM": "", "Str. Def.": "", 
        "TD Avg.": "", "TD Acc.": "", "TD Def.": "", "Sub. Avg.": "", 
        "Wins": "", "Losses": "", "Draws": ""
    }
    
    # Parse record
    record_section = soup.find("span", class_="b-content__title-record")
    if record_section:
        record_text = record_section.text.replace("Record:", "").strip()
        record_parts = record_text.split("-")
        if len(record_parts) == 3:
            stats["Wins"], stats["Losses"], stats["Draws"] = record_parts
    
    details = soup.find_all("li", class_="b-list__box-list-item")
    for detail in details:
        text = detail.text.strip()
        if "Height:" in text:
            stats["Height"] = text.split(":")[-1].strip()
        elif "Weight:" in text:
            stats["Weight"] = text.split(":")[-1].strip()
        elif "Reach:" in text:
            stats["Reach"] = text.split(":")[-1].strip()
        elif "STANCE:" in text:
            stats["Stance"] = text.split(":")[-1].strip()
        elif "DOB:" in text:
            stats["DOB"] = text.split(":")[-1].strip()
        elif "SLpM:" in text:
            stats["SLpM"] = text.split(":")[-1].strip()
        elif "Str. Acc.:" in text:
            stats["Str. Acc."] = text.split(":")[-1].strip()
        elif "SApM:" in text:
            stats["SApM"] = text.split(":")[-1].strip()
        elif "Str. Def:" in text:
            stats["Str. Def."] = text.split(":")[-1].strip()
        elif "TD Avg.:" in text:
            stats["TD Avg."] = text.split(":")[-1].strip()
        elif "TD Acc.:" in text:
            stats["TD Acc."] = text.split(":")[-1].strip()
        elif "TD Def.:" in text:
            stats["TD Def."] = text.split(":")[-1].strip()
        elif "Sub. Avg.:" in text:
            stats["Sub. Avg."] = text.split(":")[-1].strip()
    
    return stats

# Iterate through each letter
for letter in LETTERS:
    url = BASE_URL.format(letter)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve data for {letter}")
        continue
    
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="b-statistics__table")
    
    if table:
        rows = table.find_all("tr")[1:]  # Skip header row
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            
            fighter_name = cols[0].text.strip()
            fighter_link = cols[0].find("a")["href"] if cols[0].find("a") else ""
            
            if fighter_link:
                fighter_stats = scrape_fighter_stats(fighter_link)
                if fighter_stats:
                    fighters_data.append(fighter_stats)
    
    print(f"Scraped data for letter {letter}")
    time.sleep(2)  # Sleep to avoid overwhelming the server

# Convert to DataFrame and save to CSV
fighters_df = pd.DataFrame(fighters_data)
fighters_df.to_csv(r"C:\Users\jorda\OneDrive\Documents\GitHub\JDMmessingaround\UfcProject\ufc_fighters_detailed_stats.csv", index=False)

print("Data saved to ufc_fighters_detailed_stats.csv")


print(f'Total run time is {time.time() - start} seconds')