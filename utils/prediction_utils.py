from datetime import datetime, timedelta

from typing import Optional


def input_array(start_time: datetime):
     dt = start_time
     print(dt)
     day = dt.weekday()  # 0=Monday, 6=Sunday
     day_of_year = (dt - datetime(dt.year, 1, 1)).days + 1
     first_days_of_exam_weeks = [298, 354, 31, 94, 164, 192]
     upcoming_exam_weeks = [day for day in first_days_of_exam_weeks if day > day_of_year]
     exam_week = min(upcoming_exam_weeks, default=None)  # Get the next exam week or None
     exam_delta = exam_week - day_of_year if exam_week else None

     intervals_in_minutes = [hour * 60 + minute for hour in range(start_time.hour, 23 + 1) for minute in range(0, 60, 15)]
     print(intervals_in_minutes)
     interval_data = []
     for interval in intervals_in_minutes:
         data = {
             "createdAt": interval,
             "currentDay": day,
             "currentDayofYear": day_of_year,
             "currentExamDelta": exam_delta
         }
         interval_data.append(data)
     return interval_data


