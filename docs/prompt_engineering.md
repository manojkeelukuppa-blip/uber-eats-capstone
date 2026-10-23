# Prompt Engineering for the Uber Eats Capstone

## 1. Purpose

Prompt engineering helps generate consistent, structured, and useful synthetic data and documentation for the Uber Eats Operational & Customer Intelligence System.

The prompts below demonstrate four approaches:

1. P.T.C.F. prompting
2. Few-shot prompting
3. Role-based prompting
4. Structured-output prompting

AI-generated outputs must be validated before they are used in the project.

## 2. P.T.C.F. Prompting

P.T.C.F. stands for Persona, Task, Context, and Format.

### Example prompt

**Persona:** You are a synthetic data engineer specialising in food delivery platforms.

**Task:** Generate sample customer records for a machine learning capstone.

**Context:** The dataset will support customer segmentation and exploratory analysis. It must use synthetic information, unique customer IDs, reasonable age values, and consistent category labels.

**Format:** Return a CSV-compatible table with these columns: `customer_id`, `age`, `city`, `membership_type`, and `preferred_cuisine`. Do not include real personal information.

### Expected benefit

The four explicit sections clarify the role, objective, intended use, and output format. This makes the request easier to evaluate and reproduce.

## 3. Few-Shot Prompting

Few-shot prompting provides examples to demonstrate the expected output pattern.

### Example prompt

Generate three synthetic restaurant records using the same structure and conventions as these examples:

| restaurant_id | restaurant_name | cuisine | average_rating |
|---:|---|---|---:|
| 1 | Restaurant_001 | Indian | 4.20 |
| 2 | Restaurant_002 | Chinese | 3.80 |

Return three new records with unique IDs. Use restaurant names in the format `Restaurant_NNN`, cuisine categories from Indian, Chinese, Italian, Mexican, and Fast Food, and ratings between 3.0 and 5.0.

Return only the table with the same four columns.

### Expected benefit

Examples communicate naming conventions, column order, data types, and value formats. They help reduce inconsistent outputs.

## 4. Role-Based Prompting

Role-based prompting assigns a domain-specific role to guide the response.

### Example prompt

Act as a senior data analyst working on a food delivery platform.

Review the following hypothetical operational questions:

- Which order patterns may indicate peak demand?
- Which delivery variables may affect delivery duration?
- What customer and restaurant characteristics may support segmentation?

For each question, recommend relevant dataset columns, an appropriate analytical method, and one limitation to consider. Clearly distinguish hypotheses from findings, and do not claim to have analysed data that has not been provided.

### Expected benefit

A defined role encourages responses that are relevant to the intended domain and analytical task. It does not guarantee that the response is correct, so outputs must still be checked.

## 5. Structured-Output Prompting

Structured-output prompting specifies an exact schema for the response.

### Example prompt

Create a synthetic order record using this schema:

- `order_id`: positive integer
- `customer_id`: positive integer
- `restaurant_id`: positive integer
- `order_value`: positive numeric value
- `distance_km`: positive numeric value
- `order_status`: one of Delivered, Cancelled, or Failed

Return valid JSON with exactly these keys and compatible value types. Do not add commentary or extra fields. Use only synthetic values.

### Example expected output

```json
{
  "order_id": 1001,
  "customer_id": 27,
  "restaurant_id": 12,
  "order_value": 425.50,
  "distance_km": 3.8,
  "order_status": "Delivered"
}
```

### Expected benefit

A defined schema makes output easier to parse and validate in automated workflows. Valid JSON alone does not guarantee that referenced IDs exist or that business rules are satisfied.

## 6. Responsible Use and Validation

All AI-generated or synthetic outputs should be reviewed before use.

Validation should include:

- Required columns and data types
- Missing and duplicate values
- Allowed category values
- Primary-key uniqueness
- Foreign-key consistency
- Plausible numeric ranges
- Consistency between related datasets

Synthetic records must not be represented as actual customer, restaurant, driver, or transaction records from Uber Eats.

## 7. Conclusion

P.T.C.F., few-shot, role-based, and structured-output prompting address different aspects of prompt design. Combining clear instructions with examples, domain context, output schemas, and programmatic validation can improve consistency and usefulness in the capstone workflow.
