# # Slaughter House
# import random
# from selenium import webdriver
# import time
# import os
# from selenium.webdriver.common.by import By


# from webdriver_manager.chrome import ChromeDriverManager
# from selenium import webdriver


# # The URLs of the websites you want to capture and the CSS selector of
# # the element
# # websites = [{'url': 'https://www.youtube.com/live/bkvaeX5EoUs?feature=share',
# # 'selector': '#cinematics'},

# website = {
#     "url": "http://www.webbkameror.se/byggkameror/stockholmsstad/stockholm_5_1280.php",
#     "selector": ".bild-border",
# }

# # Create a directory for the website if it doesn't exist
# directory = "/home/sofia_afn/Documents/thesis_data/Slaughter_House_area2"
# if not os.path.exists(directory):
#     os.makedirs(directory)

# # Counter for memory management
# counter = 0


# chrome_driver_path = ChromeDriverManager().install()

# # Initialize the WebDriver with the downloaded driver
# driver = webdriver.Chrome(executable_path=chrome_driver_path)

# driver.maximize_window()  # For maximizing window
# driver.implicitly_wait(20)  # gives an implicit wait for 20 seconds

# # Counter for the number of screenshots
# counter = 0

# # The loop will run until 20 screenshots have been taken
# while counter < 30:
#     try:
#         # Navigate to the website
#         driver.get(website["url"])
#         time.sleep(5)

#         # Find the element
#         element = driver.find_element(By.CSS_SELECTOR, ".bild-border")

#         # Take screenshot of the element and save it with the current timestamp
#         timestamp = time.strftime("%d%m-%H%M")
#         element.screenshot(f"{directory}/Slaughterhouse-{timestamp}.png")
#         print(f"Screenshot taken at {timestamp}")
#         # Increment the counter
#         counter += 1
#         # Wait for a random amount of time between 1 minute and 1 hour
#         sleep_time = random.randint(60, 1200)

#         print(f"Sleeping for {sleep_time} seconds until the next screenshot.")
#         time.sleep(sleep_time)
#     except Exception as e:
#         # If an error occurs, make a beep sound and print the error
#         # beep.beep(frequency=440, secs=1, volume=100)
#         print(f"Slaughter_House_webcam: {e}")

# # Close the webdriver
# # driver.quit()

import random
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Define the URL of the website you want to capture
website = {
    "url": "http://www.webbkameror.se/byggkameror/stockholmsstad/stockholm_5_1280.php",
    "selector": ".bild-border",
}

# Create a directory for the website if it doesn't exist
directory = "/home/sofia_afn/Documents/thesis_data/Slaughter_House_area2"
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the Chrome WebDriver using WebDriver Manager
chrome_driver_path = ChromeDriverManager().install()
driver = webdriver.Chrome(executable_path=chrome_driver_path)
driver.maximize_window()
driver.implicitly_wait(20)

# Define the total number of screenshots you want to capture in 24 hours
total_screenshots = 30  # Adjust this number as needed
screenshots_taken = 0

# Set the start time and end time for the 24-hour period (in seconds since epoch)
start_time = int(time.time())
end_time = start_time + 24 * 60 * 60  # 24 hours

while screenshots_taken < total_screenshots:
    try:
        # Check if the current time is within the 24-hour period
        current_time = int(time.time())
        if current_time >= end_time:
            break  # Exit the loop if the 24-hour period is over

        # Navigate to the website
        driver.get(website["url"])
        time.sleep(5)

        # Find the element by CSS selector
        element = driver.find_element(By.CSS_SELECTOR, ".bild-border")

        # Take a screenshot and save it with the current timestamp
        timestamp = time.strftime("%d%m-%H%M")
        screenshot_filename = f"{directory}/Slaughterhouse-{timestamp}.png"
        element.screenshot(screenshot_filename)
        print(f"Screenshot taken at {timestamp}")

        # Increment the counter
        screenshots_taken += 1

        # Wait for a random amount of time between 1 minute and 1 hour
        sleep_time = random.randint(60, 3600)  # Adjust the range as needed
        print(f"Sleeping for {sleep_time} seconds until the next screenshot.")
        time.sleep(sleep_time)
    except Exception as e:
        print(f"Slaughter_House_webcam: {e}")

# Close the WebDriver
driver.quit()
