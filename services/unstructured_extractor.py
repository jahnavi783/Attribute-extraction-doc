import re
from typing import List, Dict


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _add(records: list[dict], seen: set, attribute: str, value: str):
    attribute = _clean(attribute)
    value = _clean(value)
    if not attribute or not value:
        return

    key = (attribute.lower(), value.lower())
    if key in seen:
        return

    seen.add(key)
    records.append({"attribute": attribute, "value": value})


def _first_group(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def extract_unstructured_text(text: str) -> List[Dict[str, str]]:
    """
    Generic unstructured extractor.
    It is not hardcoded to one file. It uses reusable patterns for:
    IDs, dates, amounts, phone, email, temperature, pressure, status, etc.
    """
    text = _clean(text)
    records: list[dict] = []
    seen = set()

    if not text:
        return records

    # ---------------------------------------------------------
    # 1. Strong pattern-based fields
    # ---------------------------------------------------------

    # Work order / PO / invoice / equipment / tracking
    for m in re.finditer(r"\bWO[- ]?\d+\b", text, re.I):
        _add(records, seen, "Workorder Number", m.group(0))

    for m in re.finditer(r"\bPO[- ]?\d+\b", text, re.I):
        _add(records, seen, "Purchase Order Number", m.group(0))

    for m in re.finditer(r"\bINV[- ]?\d+\b", text, re.I):
        _add(records, seen, "Invoice Number", m.group(0))

    for m in re.finditer(r"\bEQ[- ]?\d+\b", text, re.I):
        _add(records, seen, "Equipment ID", m.group(0))

    for m in re.finditer(r"\bTRK[- ]?\d+\b", text, re.I):
        _add(records, seen, "Tracking ID", m.group(0))

    # Dates
    for m in re.finditer(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text):
        _add(records, seen, "Maintenance Date", m.group(0))

    # Temperature
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:°?\s*C|celcius|celsius)\b", text, re.I):
        _add(records, seen, "Temperature", f"{m.group(1)} C")

    # Pressure
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*PSI\b", text, re.I):
        _add(records, seen, "Pressure", f"{m.group(1)} PSI")

    # Engine capacity
    for m in re.finditer(r"\b(\d{3,5})\s*CC\b", text, re.I):
        _add(records, seen, "Engine Capacity", f"{m.group(1)} CC")

    # Mileage
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*KMPL\b", text, re.I):
        _add(records, seen, "Mileage", f"{m.group(1)} KMPL")

    # Phone
    for m in re.finditer(r"\+?\d[\d\-\s]{8,}\d", text):
        _add(records, seen, "Phone Number", _clean(m.group(0)))

    # Email
    for m in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        _add(records, seen, "Email Address", m.group(0))

    # INR / amount
    for m in re.finditer(r"(₹\s?\d[\d,]*|\b\d[\d,]*\s?INR\b)", text, re.I):
        _add(records, seen, "Final Amount", _clean(m.group(0)))

    # ---------------------------------------------------------
    # 2. Phrase-based extraction
    # ---------------------------------------------------------

    # Site
    site = _first_group(
        r"\bfor\s+(Plant\s+[A-Z0-9]+|Warehouse\s+\d+|Site\s+[A-Z0-9]+)\b",
        text,
        re.I,
    )
    if site:
        _add(records, seen, "Site", site)

    # Customer name
    customer = _first_group(
        r"(?:customer|custmer full name|customer name)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+)",
        text,
        re.I,
    )
    if customer:
        _add(records, seen, "Customer Name", customer)

    # Assigned engineer / technician / account manager
    engineer = _first_group(
        r"(?:assigned enginer|assigned engineer|engineer|technician|tecnician|account mngr|account manager)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+)",
        text,
        re.I,
    )
    if engineer:
        _add(records, seen, "Assigned Engineer", engineer)

    # Machine type
    machine_type = _first_group(
        r"(?:machin type|machine type|equipment type|device type)\s*(?:is|was|:)?\s*([A-Za-z][A-Za-z0-9\-\s]{2,40})",
        text,
        re.I,
    )
    if machine_type:
        _add(records, seen, "Machine Type", machine_type)

    # Model number
    model_number = _first_group(
        r"(?:model nummber|model number|model no|model)\s*(?:recorded.*?as|appears as|is|was|:)?\s*([A-Za-z0-9\-]+)",
        text,
        re.I,
    )
    if model_number:
        _add(records, seen, "Model Number", model_number)

    # Payment method
    payment_method = _first_group(
        r"(?:payment methd|payment method|pay method|mode of payment|preffered payment methd)\s*(?:was|is|will be|:)?\s*"
        r"(bank transfer|upi|cash|credit card|debit card|net banking)",
        text,
        re.I,
    )
    if payment_method:
        _add(records, seen, "Payment Method", payment_method)

    # Payment status
    payment_status = _first_group(
        r"(?:payment stat|payment status|pay status)\s*(?:was|is|still|marked as|:)?\s*"
        r"(pending|approved|rejected|completed|closed|open|scheduled|in progress|paid|active|inactive)",
        text,
        re.I,
    )
    if payment_status:
        _add(records, seen, "Payment Status", payment_status)

    # Order status
    order_status = _first_group(
        r"(?:current ordeer status|current order status|order status)\s*(?:was|is|marked as|:)?\s*"
        r"(pending|approved|rejected|completed|closed|open|scheduled|in progress|paid|active|inactive)",
        text,
        re.I,
    )
    if order_status:
        _add(records, seen, "Order Status", order_status)

    # Delivery status
    delivery_status = _first_group(
        r"(?:delivary status|delivery status|shipping status|dispatch status)\s*(?:was|is|marked as|after|:)?\s*"
        r"(pending|approved|rejected|completed|closed|open|scheduled|in progress|active|inactive)",
        text,
        re.I,
    )
    if delivery_status:
        _add(records, seen, "Delivery Status", delivery_status)

    # Billing entity
    billing_entity = _first_group(
        r"(?:billing entitty|billing entity|billing company|bill to company)\s*(?:should remain|is|was|:)?\s*([A-Z][A-Za-z0-9&.\-\s]{2,60})",
        text,
        re.I,
    )
    if billing_entity:
        _add(records, seen, "Billing Entity", billing_entity)

    # Shipping address
    shipping_address = _first_group(
        r"(?:shpping addres|shipping address|delivery address)\s*(?:is|was|:)?\s*([A-Za-z0-9,\-\s]{5,90})",
        text,
        re.I,
    )
    if shipping_address:
        _add(records, seen, "Shipping Address", shipping_address)

    # Fuel type
    fuel_type = _first_group(
        r"(?:fuel typ|fuel type)\s*(?:for.*?\s)?(?:was|is|:)?\s*(diesel|petrol|electric|cng)",
        text,
        re.I,
    )
    if fuel_type:
        _add(records, seen, "Fuel Type", fuel_type)

    # Vehicle type
    vehicle_type = _first_group(
        r"(?:vehcle type|vehicle type)\s*(?:was|is|:)?\s*([A-Za-z][A-Za-z\s\-]{2,40})",
        text,
        re.I,
    )
    if vehicle_type:
        _add(records, seen, "Vehicle Type", vehicle_type)

    # Insurance status
    insurance_status = _first_group(
        r"(?:insurence status|insurance status)\s*(?:is|was|:)?\s*"
        r"(pending|approved|rejected|completed|closed|open|scheduled|in progress|active|inactive)",
        text,
        re.I,
    )
    if insurance_status:
        _add(records, seen, "Insurance Status", insurance_status)

    # Ownership type
    ownership_type = _first_group(
        r"(?:ownership typ|ownership type)\s*(?:is|was|:)?\s*([A-Za-z][A-Za-z\s\-]{2,40})",
        text,
        re.I,
    )
    if ownership_type:
        _add(records, seen, "Ownership Type", ownership_type)

    # Remarks
    remarks = _first_group(
        r"(?:remarks|comment)\s*(?:saying|says|is|:)?\s*['\"]?([^'\"]{5,120})",
        text,
        re.I,
    )
    if remarks:
        _add(records, seen, "Remarks", remarks)

    # Priority
    if re.search(r"\burgent\b", text, re.I):
        _add(records, seen, "Priority Level", "Urgent")

    return records