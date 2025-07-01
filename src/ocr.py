import easyocr
from dataclasses import dataclass


@dataclass
class ExtractorData:
    product_name: str
    bill_id: str
    phone: str
    address: str
    datetime: str
    sub_tprice: str
    tprice: str
    tdiscount: str
    tax: str
    discount: str
    discount_percentage: str
