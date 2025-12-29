import os
import time
import re
import requests
import argparse
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException

# --- CONFIGURATION ---
# TARGET_URL 1 = "https://webcoos.org/cameras/oakisland_east/?gallery=oakisland_east-one-minute-stills-s3" # <--- PASTE YOUR URL HERE
# TARGET_URL 2 = "https://webcoos.org/cameras/newport_shilo/"
TARGET_URL = "https://webcoos.org/cameras/currituck_hampton_inn/?gallery=currituck_hampton_inn-one-minute-stills-s3"
DOWNLOAD_FOLDER = r"C:\Users\User\source\repos\SwellSight.AI\data\real\images"

# --- DATE CONFIGURATION ---
START_DATE = "2024-01-15"  # Format: YYYY-MM-DD, or None for today
DAYS_TO_SCRAPE = 30        # How many days back to go from start date

# --- DIVERSITY SETTINGS ---
MAX_IMAGES_PER_DATE = 20  # Stop after 20 images per date
DATE_STEP = 4             # Skip 4 days between checks

# --- XPATHS ---
XPATH_OPEN_DATE_PICKER = "/html/body/div/div/main/section/div/div[2]/ul/li[1]/div/div/div/button"
XPATH_CALENDAR_CONTAINER = "//div[contains(@class, 'origin-top-right') and contains(@class, 'absolute')]"
XPATH_CALENDAR_TABLE = f"{XPATH_CALENDAR_CONTAINER}//table"
XPATH_CALENDAR_BODY = f"{XPATH_CALENDAR_TABLE}/tbody"
XPATH_PREV_MONTH_BTN = f"{XPATH_CALENDAR_CONTAINER}//button[1]"
XPATH_CURRENT_MONTH_YEAR = f"{XPATH_CALENDAR_CONTAINER}//h2"
XPATH_FIRST_THUMBNAIL = "/html/body/div/div/main/section/div/div[2]/div/div/div[1]/div[1]/div[1]/img[2]"
XPATH_NEXT_ARROW_BTN = "/html/body/div/div/main/section/div/div[2]/div/div/div[2]/div/div[3]"
XPATH_MODAL_CONTAINER = "/html/body/div/div/main/section/div/div[2]/div/div/div[2]"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Beach camera data scraper')
    parser.add_argument('--start-date', type=str, 
                       help='Start date in YYYY-MM-DD format (e.g., 2024-01-15)')
    parser.add_argument('--days-back', type=int, default=30,
                       help='Number of days to go back from start date (default: 30)')
    return parser.parse_args()

def navigate_to_month(driver, wait, target_date):
    """Navigate to the month containing the target date"""
    max_attempts = 24  # Don't go back more than 2 years
    attempts = 0
    
    while attempts < max_attempts:
        try:
            # Get current month/year displayed
            month_year_elem = wait.until(EC.presence_of_element_located((By.XPATH, XPATH_CURRENT_MONTH_YEAR)))
            current_month_year = month_year_elem.text.strip()
            print(f"  Current calendar shows: {current_month_year}")
            
            # Parse current month/year (assuming format like "November 2024")
            parts = current_month_year.split()
            if len(parts) != 2:
                print(f"  Unexpected month/year format: {current_month_year}")
                return False
                
            current_month_name, current_year = parts
            current_year = int(current_year)
            
            # Convert month name to number
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            try:
                current_month = month_names.index(current_month_name) + 1
            except ValueError:
                print(f"  Unknown month name: {current_month_name}")
                return False
            
            # Check if we're at the target month/year
            if current_year == target_date.year and current_month == target_date.month:
                print(f"  Found target month: {target_date.strftime('%B %Y')}")
                return True
            
            # If target date is in the past, click previous month
            if (target_date.year < current_year) or (target_date.year == current_year and target_date.month < current_month):
                print(f"  Need to go back from {current_month_name} {current_year} to {target_date.strftime('%B %Y')}")
                
                # Try multiple selectors for the previous month button
                prev_btn = None
                selectors = [
                    f"{XPATH_CALENDAR_CONTAINER}//button[1]",
                    f"{XPATH_CALENDAR_CONTAINER}//svg[contains(@class, 'chevron-left')]/parent::button",
                    f"{XPATH_CALENDAR_CONTAINER}//button[contains(.//polyline, '15 6 9 12 15 18')]"
                ]
                
                for i, selector in enumerate(selectors):
                    try:
                        prev_btn = driver.find_element(By.XPATH, selector)
                        print(f"  Found prev button with selector {i+1}")
                        break
                    except NoSuchElementException:
                        continue
                
                if prev_btn:
                    driver.execute_script("arguments[0].click();", prev_btn)
                    time.sleep(2)  # Increased wait time
                    attempts += 1
                else:
                    print("  Could not find previous month button")
                    return False
            else:
                # Target date is in the future, can't navigate forward
                print(f"  Target date {target_date.strftime('%Y-%m-%d')} is in the future relative to {current_month_name} {current_year}")
                return False
                
        except Exception as e:
            print(f"  Error navigating to month: {e}")
            return False
    
    print(f"  Could not navigate to {target_date.strftime('%Y-%m')} after {max_attempts} attempts")
    return False

def find_date_in_calendar_simple(driver, wait, target_date):
    """
    Simplified approach to find and click a date in the calendar
    """
    try:
        print(f"  Opening date picker...")
        
        # Open date picker using CSS selector
        picker = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button")))
        picker.click()
        time.sleep(3)
        
        print(f"  Looking for date {target_date.day}...")
        
        # Try to find the date using multiple approaches
        day_str = str(target_date.day)
        
        # Approach 1: Look for p tags with the day number
        day_elements = driver.find_elements(By.XPATH, f"//p[text()='{day_str}']")
        
        for elem in day_elements:
            try:
                # Check if this element is in a calendar context
                parent = elem.find_element(By.XPATH, "..")
                if "cursor-pointer" in parent.get_attribute("class") or "button" in parent.get_attribute("role"):
                    print(f"  Found clickable date element for day {day_str}")
                    return parent
            except:
                continue
        
        # Approach 2: Look for divs with role="button" containing the day
        button_elements = driver.find_elements(By.CSS_SELECTOR, "div[role='button']")
        
        for elem in button_elements:
            try:
                if elem.text.strip() == day_str:
                    print(f"  Found button element for day {day_str}")
                    return elem
            except:
                continue
        
        print(f"  Could not find day {day_str} in calendar")
        return None
        
    except Exception as e:
        print(f"  Error in simplified date finder: {e}")
        return None

def generate_date_range(start_date, days_back, date_step):
    """Generate a list of dates to scrape, going backwards from start_date"""
    dates = []
    current_date = start_date
    
    for i in range(0, days_back, date_step):
        dates.append(current_date - timedelta(days=i))
    
    return dates

def setup_driver():
    options = webdriver.ChromeOptions()
    # Basic stability options only
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Don't disable JavaScript - the site needs it
    # options.add_argument("--headless")  # Comment out headless for now
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(5)
        return driver
    except Exception as e:
        print(f"Failed to create Chrome driver: {e}")
        print("Make sure ChromeDriver is installed and in your PATH")
        return None

def get_next_index(folder):
    if not os.path.exists(folder): return 1
    max_idx = 0
    pattern = re.compile(r"beach_(\d+)\.jpg")
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > max_idx: max_idx = idx
    return max_idx + 1

def download_image(img_url, folder, current_index):
    try:
        if not img_url: return False
        filename = f"beach_{str(current_index).zfill(3)}.jpg"
        filepath = os.path.join(folder, filename)
        
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"  [SAVED] {filename}")
            return True
        return False
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def run_slideshow(driver, wait, folder, global_counter):
    seen_urls = set()
    images_today = 0
    
    try:
        wait.until(EC.visibility_of_element_located((By.XPATH, XPATH_MODAL_CONTAINER)))
    except TimeoutException:
        print("  Error: Gallery modal did not open.")
        return global_counter

    while images_today < MAX_IMAGES_PER_DATE:
        try:
            img_element = wait.until(EC.visibility_of_element_located((By.XPATH, f"{XPATH_MODAL_CONTAINER}//img")))
            src = img_element.get_attribute('src')
            
            if src in seen_urls:
                print("  [INFO] Slideshow looped. Ending this date.")
                break
            seen_urls.add(src)

            if download_image(src, folder, global_counter):
                global_counter += 1
                images_today += 1

            if images_today >= MAX_IMAGES_PER_DATE:
                print(f"  [LIMIT] Reached {MAX_IMAGES_PER_DATE} images for this date. Moving on.")
                break

            next_arrow = driver.find_element(By.XPATH, XPATH_NEXT_ARROW_BTN)
            if "disabled" in next_arrow.get_attribute("class"):
                print("  [INFO] Next arrow disabled.")
                break

            driver.execute_script("arguments[0].click();", next_arrow)
            time.sleep(1.5) 

        except (NoSuchElementException, StaleElementReferenceException):
            print("  [INFO] Slideshow ended.")
            break
        except Exception as e:
            print(f"  [WARN] {e}")
            break

    return global_counter

def restart_driver(driver):
    """Restart the browser driver"""
    try:
        if driver:
            driver.quit()
    except:
        pass
    
    time.sleep(2)
    return setup_driver()

def safe_find_element(driver, by, value, timeout=5):
    """Safely find an element with timeout"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except:
        return None

def debug_page_elements(driver):
    """Debug function to see what elements are on the page"""
    try:
        print("  DEBUG: Page elements analysis:")
        
        # Check page title
        title = driver.title
        print(f"    Page title: {title}")
        
        # Check for buttons
        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"    Found {len(buttons)} buttons")
        for i, btn in enumerate(buttons[:3]):  # Show first 3
            try:
                text = btn.text.strip()[:50]  # First 50 chars
                classes = btn.get_attribute("class") or ""
                print(f"      Button {i+1}: '{text}' class='{classes[:50]}'")
            except:
                pass
        
        # Check for divs with cursor pointer
        cursor_divs = driver.find_elements(By.CSS_SELECTOR, "div[class*='cursor']")
        print(f"    Found {len(cursor_divs)} divs with cursor classes")
        
        # Check for elements with role=button
        role_buttons = driver.find_elements(By.CSS_SELECTOR, "[role='button']")
        print(f"    Found {len(role_buttons)} elements with role='button'")
        
        # Check if the specific XPath exists
        try:
            specific_element = driver.find_element(By.XPATH, XPATH_OPEN_DATE_PICKER)
            print(f"    ✓ Found element with original XPath")
            print(f"      Text: '{specific_element.text.strip()}'")
            print(f"      Class: '{specific_element.get_attribute('class')}'")
        except:
            print(f"    ✗ Could not find element with original XPath: {XPATH_OPEN_DATE_PICKER}")
            
    except Exception as e:
        print(f"    Debug error: {e}")

def main():
    args = parse_arguments()
    
    # Determine start date
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            print(f"Starting from specified date: {start_date.strftime('%Y-%m-%d')}")
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-15)")
            return
    else:
        start_date = datetime.now()
        print(f"Starting from today: {start_date.strftime('%Y-%m-%d')}")
    
    # Generate list of dates to scrape
    target_dates = generate_date_range(start_date, args.days_back, DATE_STEP)
    print(f"Will scrape {len(target_dates)} dates going back {args.days_back} days")
    
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    global_counter = get_next_index(DOWNLOAD_FOLDER)
    print(f"Starting at index: {global_counter}")

    driver = setup_driver()
    if not driver:
        print("Failed to start browser. Exiting.")
        return
    
    wait = WebDriverWait(driver, 10)

    try:
        driver.get(TARGET_URL)
        time.sleep(5)
        print("Page loaded successfully")
        
        # Debug the page on first load
        debug_page_elements(driver)

        for target_date in target_dates:
            print(f"\n--- Processing Date: {target_date.strftime('%Y-%m-%d')} ---")
            
            try:
                # Step 1: Open the date picker first
                print("  Step 1: Opening date picker...")
                try:
                    # Use the original XPath you provided
                    picker = wait.until(EC.element_to_be_clickable((By.XPATH, XPATH_OPEN_DATE_PICKER)))
                    print(f"  Found date picker button")
                    driver.execute_script("arguments[0].click();", picker)
                    time.sleep(3)
                    print("  Date picker clicked")
                    
                except Exception as picker_error:
                    print(f"  Error opening date picker with original XPath: {picker_error}")
                    
                    # Fallback: try to find any button that might open the date picker
                    print("  Trying fallback methods...")
                    try:
                        # Look for buttons with specific text patterns
                        buttons = driver.find_elements(By.TAG_NAME, "button")
                        print(f"  Found {len(buttons)} buttons on page")
                        
                        for i, button in enumerate(buttons[:5]):  # Check first 5 buttons
                            try:
                                text = button.text.strip()
                                print(f"    Button {i+1}: '{text}'")
                                
                                # If button has date-like text or is empty (might be icon button)
                                if not text or any(word in text.lower() for word in ["date", "calendar", "select", "pick"]):
                                    print(f"    Trying button {i+1}")
                                    driver.execute_script("arguments[0].click();", button)
                                    time.sleep(2)
                                    
                                    # Check if calendar appeared
                                    calendar_elements = driver.find_elements(By.XPATH, "//table | //div[contains(@class, 'calendar')] | //div[contains(@class, 'date')]")
                                    if calendar_elements:
                                        print(f"    Success! Calendar opened with button {i+1}")
                                        break
                                        
                            except Exception as button_error:
                                continue
                        else:
                            print("  Could not open date picker")
                            continue
                            
                    except Exception as fallback_error:
                        print(f"  Fallback failed: {fallback_error}")
                        continue
                
                # Step 2: Now look for the specific date
                print(f"  Step 2: Looking for day {target_date.day}...")
                day_str = str(target_date.day)
                
                # Wait a moment for calendar to load
                time.sleep(2)
                
                # Find all elements with the day number
                day_elements = driver.find_elements(By.XPATH, f"//*[text()='{day_str}']")
                print(f"  Found {len(day_elements)} elements with text '{day_str}'")
                
                clicked = False
                for i, elem in enumerate(day_elements):
                    try:
                        # Check if this looks like a calendar date
                        parent = elem.find_element(By.XPATH, "..")
                        class_attr = parent.get_attribute("class") or ""
                        
                        print(f"    Element {i+1}: class='{class_attr}'")
                        
                        if any(keyword in class_attr.lower() for keyword in ["cursor", "button", "click", "date"]):
                            print(f"    Clicking on date element {i+1}")
                            driver.execute_script("arguments[0].click();", parent)
                            time.sleep(3)
                            clicked = True
                            break
                    except Exception as elem_error:
                        print(f"    Error with element {i+1}: {elem_error}")
                        continue
                
                if not clicked:
                    print(f"  Could not find clickable date for {day_str}")
                    continue
                
                # Step 3: Try to enter slideshow
                print("  Step 3: Looking for images...")
                try:
                    print("  Looking for first thumbnail...")
                    first_thumb = driver.find_element(By.XPATH, XPATH_FIRST_THUMBNAIL)
                    print("  Found thumbnail, clicking...")
                    driver.execute_script("arguments[0].click();", first_thumb)
                    
                    global_counter = run_slideshow(driver, wait, DOWNLOAD_FOLDER, global_counter)
                    
                    print("  Refreshing page...")
                    driver.refresh()
                    time.sleep(3)
                    
                except NoSuchElementException:
                    print(f"  No images found for {target_date.strftime('%Y-%m-%d')}")
                    driver.refresh()
                    time.sleep(3)

            except Exception as e:
                print(f"  Error processing {target_date.strftime('%Y-%m-%d')}: {e}")
                driver.refresh()
                time.sleep(3)
                continue

    finally:
        if driver:
            driver.quit()
        print("Done.")

if __name__ == "__main__":
    main()