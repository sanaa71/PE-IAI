import json

# Load rules from JSON file
with open("rules.json") as file:
    data = json.load(file)
    rules = data["rules"]

print("🧠 Welcome to the Enhanced Rule-Based Expert System")
print("Please enter your symptoms (comma separated):")

user_input = input("👉 Symptoms: ").lower().split(",")
symptoms = [s.strip() for s in user_input if s.strip()]

# Store best match
best_rule = None
best_score = 0.0

# Evaluate rules
for rule in rules:
    required = rule["if"]
    matched = [s for s in required if s in symptoms]
    score = len(matched) / len(required)  # confidence = matched / total

    if score > best_score:
        best_score = score
        best_rule = rule

# Display results
if best_score == 1.0:
    print(f"\n💡 Diagnosis: {best_rule['then']}")
    print(f"💊 Treatment: {best_rule['treatment']}")
elif best_score >= 0.5:
    print(f"\n🤔 Possible Diagnosis: {best_rule['then']}")
    print(f"Confidence: {best_score*100:.0f}%")
    print(f"💊 Suggested Treatment: {best_rule['treatment']}")
else:
    print("\n⚠️ Sorry, no strong match found. Please consult a doctor.")

