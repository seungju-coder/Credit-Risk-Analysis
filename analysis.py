import pandas as pd

# 데이터 불러오기
df = pd.read_csv("data/credit_risk_dataset.csv")

# 데이터 앞부분 확인
print(df.head())

# 데이터 기본 정보 확인
print(df.info())

# 결측치 확인
print("\n[결측치 확인]")
print(df.isnull().sum())

# 수치형 데이터의 기본 통계 확인
print("\n[기초 통계]")    
print(df.describe())

# 대출 상태별 고객 수 확인
print("\n[대출 상태별 고객 수]")
print(df["loan_status"].value_counts())

# 대출 상태별 비율 확인
print("\n[대출 상태별 비율]")
print(df["loan_status"].value_counts(normalize=True))

# 대출 상태별 주요 수치 비교
print("\n[대출 상태별 평균 비교]")
print(
    df.groupby("loan_status")[
        ["person_age", "person_income", "loan_amnt",
         "loan_int_rate", "loan_percent_income",
         "person_emp_length", "cb_person_cred_hist_length"]
    ].mean()
)

import matplotlib.pyplot as plt

# 대출 상태별 고객 수
status_counts = df["loan_status"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(["Normal (0)", "Risk (1)"], status_counts.values)

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Customers")

plt.savefig("loan_status_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 소득을 4개 구간으로 나누기
df["income_group"] = pd.qcut(
    df["person_income"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)

# 소득 구간별 위험률 계산
income_risk_rate = (
    df.groupby("income_group", observed=True)["loan_status"].mean() * 100
)

print("\n[소득 구간별 위험률]")
print(income_risk_rate)

# 대출 부담률을 4개 구간으로 나누기
df["loan_burden_group"] = pd.qcut(
    df["loan_percent_income"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)

# 대출 부담률별 위험률 계산
loan_burden_risk_rate = (
    df.groupby("loan_burden_group", observed=True)["loan_status"].mean() * 100
)

print("\n[대출 부담률별 위험률]")
print(loan_burden_risk_rate)

# 대출 목적별 위험률 계산
loan_intent_risk_rate = (
    df.groupby("loan_intent")["loan_status"].mean() * 100
)

# 위험률이 높은 순서대로 정렬
loan_intent_risk_rate = loan_intent_risk_rate.sort_values(ascending=False)

print("\n[대출 목적별 위험률]")
print(loan_intent_risk_rate)

# 주택 소유 형태별 위험률 계산
home_ownership_risk_rate = (
    df.groupby("person_home_ownership")["loan_status"].mean() * 100
)

# 위험률이 높은 순서대로 정렬
home_ownership_risk_rate = home_ownership_risk_rate.sort_values(
    ascending=False
)

print("\n[주택 소유 형태별 위험률]")
print(home_ownership_risk_rate)

# 대출 등급별 위험률 계산
loan_grade_risk_rate = (
    df.groupby("loan_grade")["loan_status"].mean() * 100
)

# 위험률이 높은 순서대로 정렬
loan_grade_risk_rate = loan_grade_risk_rate.sort_values(
    ascending=False
)

print("\n[대출 등급별 위험률]")
print(loan_grade_risk_rate)

# 소득 구간별 위험률 시각화
plt.figure(figsize=(7, 4))

plt.bar(
    income_risk_rate.index,
    income_risk_rate.values
)

plt.title("Risk Rate by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Risk Rate (%)")

plt.ylim(0, 50)

# 막대 위에 위험률 숫자 표시
for i, value in enumerate(income_risk_rate.values):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.savefig("income_risk_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# 대출 부담률별 위험률 시각화
plt.figure(figsize=(7, 4))

plt.bar(
    loan_burden_risk_rate.index,
    loan_burden_risk_rate.values
)

plt.title("Risk Rate by Loan Burden Group")
plt.xlabel("Loan Burden Group")
plt.ylabel("Risk Rate (%)")

plt.ylim(0, 55)

# 막대 위에 위험률 표시
for i, value in enumerate(loan_burden_risk_rate.values):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.savefig("loan_burden_risk_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# 대출 목적별 위험률 시각화
plt.figure(figsize=(9, 5))

plt.bar(
    loan_intent_risk_rate.index,
    loan_intent_risk_rate.values
)

plt.title("Risk Rate by Loan Intent")
plt.xlabel("Loan Intent")
plt.ylabel("Risk Rate (%)")

plt.ylim(0, 35)

# 막대 위에 위험률 표시
for i, value in enumerate(loan_intent_risk_rate.values):
    plt.text(
        i,
        value + 0.8,
        f"{value:.1f}%",
        ha="center"
    )

plt.xticks(rotation=20)

plt.savefig("loan_intent_risk_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# 주택 소유 형태별 위험률 시각화
plt.figure(figsize=(7, 4))

plt.bar(
    home_ownership_risk_rate.index,
    home_ownership_risk_rate.values
)

plt.title("Risk Rate by Home Ownership")
plt.xlabel("Home Ownership")
plt.ylabel("Risk Rate (%)")

plt.ylim(0, 40)

# 막대 위에 위험률 표시
for i, value in enumerate(home_ownership_risk_rate.values):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.savefig("home_ownership_risk_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# 대출 등급별 위험률 시각화
plt.figure(figsize=(8, 5))

plt.bar(
    loan_grade_risk_rate.index,
    loan_grade_risk_rate.values
)

plt.title("Risk Rate by Loan Grade")
plt.xlabel("Loan Grade")
plt.ylabel("Risk Rate (%)")

plt.ylim(0, 105)

# 막대 위에 위험률 표시
for i, value in enumerate(loan_grade_risk_rate.values):
    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.savefig("loan_grade_risk_rate.png", dpi=300, bbox_inches="tight")
plt.show()