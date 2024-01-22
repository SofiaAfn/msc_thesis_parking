# Varbergs torg

from selenium import webdriver
import time
import os
from selenium.webdriver.common.by import By

# from webdriver_manager.chrome import ChromeDriverManager


# The URLs of the websites you want to capture and the CSS selector of
# the element
# websites = [{'url': 'https://www.youtube.com/live/bkvaeX5EoUs?feature=share',
# 'selector': '#cinematics'},

website = {
    "url": "https://www.youtube.com/watch?v=bkvaeX5EoUs&ab_channel=MED-Mediateknik",
    "selector": "#cinematics",
}

# Create a directory for the website if it doesn't exist
directory = "/home/sofia_afn/Documents/thesis_data/Varbergs_cam"
if not os.path.exists(directory):
    os.makedirs(directory)

# Counter for memory management
counter = 0

# Initialize the Chrome webdriver
driver = webdriver.Chrome("/chromedriver")

driver.maximize_window()  # For maximizing window
driver.implicitly_wait(10)  # gives an implicit wait for 20 seconds

# The loop will run indefinitely
while True:
    try:
        # Navigate to the website
        driver.get(website["url"])
        time.sleep(5)

        # Find the element
        element = driver.find_element(By.CSS_SELECTOR, "#cinematics")

        # Take screenshot of the element and save it with the current timestamp
        timestamp = time.strftime("%d%m-%H%M")
        element.screenshot(f"{directory}/Varb_{timestamp}.png")

        # Close the webdriver every 100 iterations to free up memory
        counter += 1
        if counter % 100 == 0:
            driver.quit()

        # Wait for 5 minutes (300 seconds)
        time.sleep(1800)
    except Exception as e:
        # If an error occurs, make a beep sound and print the error
        # beep.beep(frequency=440, secs=1, volume=100)
        print(f"Varbergs_torg_webcam: {e}")

# Close the webdriver
# driver.quit()
