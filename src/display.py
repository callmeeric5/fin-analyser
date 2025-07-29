import streamlit as st
import pandas as pd


def display(data):
    doc_type = (data.get("document_type") or "").lower()

    st.markdown(
        f"### 🧾 Document Type: {data.get('document_type','').capitalize() or '-'}"
    )

    def get_filtered_df(items, min_cols):
        df = pd.DataFrame(items)
        keep_cols = set(min_cols)
        for col in df.columns:
            if col in keep_cols:
                continue
            if df[col].notna().any() and df[col].astype(str).str.strip().ne("").any():
                keep_cols.add(col)
        return df[list(keep_cols)]

    if "receipt" in doc_type:
        st.markdown("#### 🏪 Store Info")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Store Name:** {data.get('store_name','') or '-'}")
            st.markdown(f"**Date:** {data.get('date','') or '-'}")
            st.markdown(f"**Time:** {data.get('time','') or '-'}")
        with col2:
            st.markdown(f"**Store Address:** {data.get('store_address','') or '-'}")
            st.markdown(f"**Store Phone:** {data.get('store_phone','') or '-'}")

        items = data.get("items", [])
        if items:
            st.markdown("#### 🛒 Items Purchased")
            df = get_filtered_df(items, ["item_name", "item_value"])
            df = df.rename(
                columns={
                    "item_name": "Name",
                    "item_desc": "Description",
                    "item_key": "Item Key",
                    "item_quantity": "Qty",
                    "item_net_price": "Net Price",
                    "item_value": "Value",
                    "item_net_worth": "Net Worth",
                    "item_vat": "VAT",
                    "item_gross_worth": "Gross Worth",
                }
            )
            st.dataframe(df.fillna(""), hide_index=True, use_container_width=True)
        else:
            st.info("No items found.")

        st.markdown("#### 💵 Payment Summary")
        st.markdown(f"**Subtotal:** {data.get('subtotal','') or '-'}")
        st.markdown(f"**Tax:** {data.get('tax','') or '-'}")
        st.markdown(f"**Tips:** {data.get('tips','') or '-'}")
        st.markdown(f"**Total:** {data.get('total','').split()[0] or '-'}")

    elif "invoice" in doc_type:
        st.markdown("#### 🏢 Parties Info")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Seller:** {data.get('seller','') or '-'}")
            st.markdown(f"**Seller Address:** {data.get('store_address','') or '-'}")
            st.markdown(f"**Invoice No:** {data.get('invoice_no','') or '-'}")
        with col2:
            st.markdown(f"**Client:** {data.get('client','') or '-'}")
            st.markdown(f"**IBAN:** {data.get('iban','') or '-'}")
            st.markdown(f"**Invoice Date:** {data.get('invoice_date','') or '-'}")

        items = data.get("items", [])
        if items:
            st.markdown("#### 📦 Invoice Items")
            df = get_filtered_df(items, ["item_name", "item_quantity", "item_value"])
            df = df.rename(
                columns={
                    "item_name": "Name",
                    "item_desc": "Description",
                    "item_key": "Item Key",
                    "item_quantity": "Qty",
                    "item_net_price": "Net Price",
                    "item_value": "Value",
                    "item_net_worth": "Net Worth",
                    "item_vat": "VAT",
                    "item_gross_worth": "Gross Worth",
                }
            )
            st.dataframe(df.fillna(""), hide_index=True, use_container_width=True)
        else:
            st.info("No items found.")

        st.markdown("#### 💰 Invoice Totals")
        st.markdown(f"**Subtotal:** {data.get('subtotal','') or '-'}")
        st.markdown(f"**Total Net Worth:** {data.get('total_net_worth','') or '-'}")
        st.markdown(f"**Total VAT:** {data.get('total_vat','') or '-'}")
        st.markdown(f"**Total Gross Worth:** {data.get('total_gross_worth','') or '-'}")
        st.markdown(f"**Tax:** {data.get('tax','') or '-'}")
        st.markdown(f"**Total:** {data.get('total','') or '-'}")

    else:
        st.markdown("#### 📄 Document Info")
        for k, v in data.items():
            if k == "items":
                continue
            st.markdown(f"**{k.replace('_',' ').capitalize()}:** {v or '-'}")

        items = data.get("items", [])
        if items:
            st.markdown("#### Items")
            df = get_filtered_df(items, ["item_name", "item_quantity", "item_value"])
            st.dataframe(df.fillna(""), hide_index=True, use_container_width=True)
