from selenium import webdriver
import os
import time
import beep

# The URL of the website you want to capture and the CSS selector of the element
website = {'url': 'http://example.com', 'selector': '#element'}

# Create a directory for the website if it doesn't exist
directory = website['url'].replace('http://', '').replace('https://', '').replace('.', '_')
if not os.path.exists(directory):
    os.makedirs(directory)

# Counter for memory management
counter = 0

# The loop will run indefinitely
while True:
    try:
        # Initialize the Firefox webdriver
        driver = webdriver.Firefox(executable_path='/path/to/geckodriver')

        # Navigate to the website
        driver.get(website['url'])
        
        # Find the element
        element = driver.find_element_by_css_selector(website['selector'])
        
        # Take screenshot of the element and save it with the current timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        element.screenshot(f'{directory}/screenshot-{timestamp}.png')
        
        # Close the webdriver every 100 iterations to free up memory
        counter += 1
        if counter % 100 == 0:
            driver.quit()
        
        # Wait for 5 minutes (300 seconds)
        time.sleep(300)
    except Exception as e:
        # If an error occurs, make a beep sound and print the error
        beep.beep(frequency=440, secs=1, volume=100)
        print(f'An error occurred: {e}')
        # Close the webdriver if an error occurs
        driver.quit()

# Close the webdriver
driver.quit()
