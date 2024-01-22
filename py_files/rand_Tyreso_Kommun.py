import random
from selenium import webdriver
import time
import os
from selenium.webdriver.common.by import By

# The URLs of the websites you want to capture and the CSS selector of the element
website = {
    "url": "http://www.webbkameror.se/byggkameror/tyresokommun/tyreso_1_1280.php",
    "selector": ".bild-border",
}

# Create a directory for the website if it doesn't exist
directory = "/home/sofia_afn/Documents/thesis_data/Tyreso_Kommun"
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the Chrome webdriver
driver = webdriver.Chrome("/chromedriver")

driver.maximize_window()  # For maximizing window
driver.implicitly_wait(10)  # gives an implicit wait for 20 seconds

# Counter for the number of screenshots
counter = 0

# The loop will run until 20 screenshots have been taken
while counter < 20:
    try:
        # Navigate to the website
        driver.get(website["url"])
        time.sleep(5)

        # Find the element
        element = driver.find_element(By.CSS_SELECTOR, ".bild-border")

        # Take screenshot of the element and save it with the current timestamp
        timestamp = time.strftime("%d%m-%H%M")
        element.screenshot(f"{directory}/Tyreso_K-{timestamp}.png")
        print(f"Screenshot taken at {timestamp}")
        # Increment the counter
        counter += 1
        # Wait for a random amount of time between 1 minute and 1 hour
        sleep_time = random.randint(60, 3600)

        print(f"Sleeping for {sleep_time} seconds until the next screenshot.")
        time.sleep(sleep_time)
    except Exception as e:
        # If an error occurs, print the error
        print(f"Tyreso_Kommun_webcam: {e}")

# Close the webdriver
driver.quit()
