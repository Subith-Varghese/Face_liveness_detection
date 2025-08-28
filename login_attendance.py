from datetime import datetime
import csv
# Attendance Logger Function
def log_attendance(name, log_file="attendance.csv"):
    now = datetime.now()  # Get current date and time
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")  # Format it as a string

    with open(log_file, mode='a', newline='') as file:  # Open file in append mode
        writer = csv.writer(file)
        writer.writerow([name, timestamp])  # Write name and timestamp to a new row

    print(f"✅ Attendance logged for {name} at {timestamp}")
