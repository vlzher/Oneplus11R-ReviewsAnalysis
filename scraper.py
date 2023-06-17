import time
from datetime import datetime
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

reviewList = []


def extract_additional_data(review):
    data_string = review.find_element(By.XPATH, './/a[@data-hook="format-strip"]').text
    pattern = r"Size:\s*(\d+)GB\s+RAM,\s*(\d+)GB\s+StorageColour:\s+(.*)"
    match = re.search(pattern, data_string)
    ram_size = int(match.group(1))
    storage_capacity = int(match.group(2))
    color = match.group(3)
    result = {
        "ram": ram_size,
        "storage": storage_capacity,
        "color": color
    }
    return result


def checkIfVerified(review):
    result = {}
    try:
        review.find_element(By.XPATH, './/span[@data-hook="avp-badge"]')
        result["verified"] = True
        return result
    except:
        result["verified"] = False
        return result


def extractStars(review):
    rating_element = review.find_element(By.XPATH, './/i[@data-hook="review-star-rating"]')
    rating_span = rating_element.find_element(By.XPATH, './span[@class="a-icon-alt"]')
    return rating_span.get_attribute("innerHTML").strip()[:3]


def extract_date(review):
    date_element = review.find_element(By.XPATH, './/span[@data-hook="review-date"]').text
    date_str = date_element.split('on ')[1]

    date_obj = datetime.strptime(date_str, '%d %B %Y')
    formatted_date = date_obj.strftime('%d/%m/%Y')
    return formatted_date


def clean_text(str):
    return re.sub(r'[^\w\s\']', ' ', str)


def extract_review(driver):
    reviews = driver.find_elements(By.XPATH, '//div[@data-hook="review"]')

    for review in reviews:
        additional_data = extract_additional_data(review)
        additional_data.update(checkIfVerified(review))
        review_dict = {
            "title": clean_text(review.find_element(By.XPATH, './/a[@data-hook="review-title"]').text),
            "body": clean_text(review.find_element(By.XPATH, './/span[@data-hook="review-body"]').text),
            "rating": extractStars(review),
            "date": extract_date(review)
        }
        review_dict.update(additional_data)
        reviewList.append(review_dict)


def main():
    product_url = "https://www.amazon.in/OnePlus-Galactic-Silver-128GB-Storage/dp/B0BSNQ2KXF/"
    review_url = product_url.replace("dp", "product-reviews") + "?pageNumber="
    driver = webdriver.Chrome()
    driver.get(review_url + str(1))
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[@data-hook="review"]'))
    )
    page_num = 108
    print("Number of Pages : ", page_num)

    for page in range(1, page_num + 1):
        print(f"Running for Page {page}📝📝📝....")
        page_url = review_url + str(page)
        print(page_url)

        driver.get(page_url)
        time.sleep(1)
        extract_review(driver)

    driver.quit()
    df = pd.DataFrame(reviewList)
    df.to_excel('data.xlsx', index=False)


if __name__ == '__main__':
    main()
