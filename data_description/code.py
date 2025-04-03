import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv("Student_performance_data.csv")
df = pd.DataFrame(Data)
print(df)

print(df.info())
print(df.describe())

plt.hist(df['StudyTimeWeekly'], bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Study Time Weekly')
plt.ylabel('Frequency')
plt.title('Histogram of Study Time Weekly')
plt.show()

plt.hist(df['GPA'], bins=10, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('GPA')
plt.ylabel('Frequency')
plt.title('Histogram of GPA')
plt.show()
