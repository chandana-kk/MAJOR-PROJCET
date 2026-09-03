"""
EnergyPulse Feature Manager
===========================
Integration point for the three new features:
1. Family Member Management
2. Automated Notifications  
3. Grounded Q&A Chatbot

This module provides UI helpers and orchestration for the new features.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any

from db import get_db
from notifications import get_notification_service
from chatbot import get_chatbot
from i18n import T


def render_family_management_ui():
    """Render the Family & Household section in settings."""
    if "auth" not in st.session_state or not st.session_state.auth.get("logged_in"):
        return
    
    db = get_db()
    email = st.session_state.auth.get("email")
    user = db.get_user(email)
    
    if not user:
        st.error("User not found")
        return
    
    household_id = user.get("household_id")
    
    st.markdown("## 👨‍👩‍👧‍👦 Family & Household")
    st.markdown("Add family members and manage their notification preferences.")
    
    # Get current family members
    family_members = db.get_family_members(household_id)
    
    # Display existing members
    if family_members:
        st.markdown("### Existing Members")
        for member in family_members:
            with st.expander(f"{member['name']} ({member.get('relationship', 'Family')})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Email:** {member['email']}")
                    if member.get('phone'):
                        st.markdown(f"**Phone:** {member['phone']}")
                    st.markdown(f"**Language:** {member.get('preferred_language', 'en')}")
                    st.markdown(f"**Notifications:**")
                    st.markdown(f"  - Bill Alerts: {'✓' if member.get('notification_bill_alerts') else '✗'}")
                    st.markdown(f"  - Optimization Tips: {'✓' if member.get('notification_optimization_tips') else '✗'}")
                
                with col2:
                    if st.button("Remove", key=f"remove_member_{member['id']}"):
                        db.remove_family_member(member['id'], household_id)
                        st.success(f"Removed {member['name']}")
                        st.rerun()
    else:
        st.info("No family members added yet.")
    
    # Add new member form
    st.markdown("### Add Family Member")
    with st.form("add_family_member_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            member_name = st.text_input("Name", placeholder="John Doe")
            relationship = st.selectbox("Relationship", ["Spouse", "Parent", "Child", "Sibling", "Other"])
            email = st.text_input("Email Address", placeholder="john@example.com")
        
        with col2:
            phone = st.text_input("Phone (optional)", placeholder="+91-9999999999")
            language = st.selectbox("Preferred Language", ["English", "हिन्दी", "ಕನ್ನಡ", "తెలుగు"],
                                   index=0)
            lang_code = {"English": "en", "हिन्दी": "hi", "ಕನ್ನಡ": "kn", "తెలుగు": "te"}.get(language, "en")
        
        notify_bills = st.checkbox("Opt-in to bill alerts", value=True)
        notify_tips = st.checkbox("Opt-in to optimization tips", value=True)
        
        if st.form_submit_button("Add Member", type="primary"):
            db_instance = get_db()
            success, msg = db_instance.add_family_member(
                household_id=household_id,
                name=member_name,
                relationship=relationship,
                email=email,
                phone=phone,
                preferred_language=lang_code,
                notify_bills=notify_bills,
                notify_tips=notify_tips
            )
            
            if success:
                st.success(f"Added {member_name} to family!")
                st.rerun()
            else:
                error_messages = {
                    "invalid_email": "Invalid email format",
                    "duplicate_email": "This email is already in the household",
                    "duplicate_member": "This member already exists"
                }
                st.error(error_messages.get(msg, "Failed to add member"))


def render_notifications_ui():
    """Render the Notifications & Alerts section in settings."""
    if "auth" not in st.session_state or not st.session_state.auth.get("logged_in"):
        return
    
    email = st.session_state.auth.get("email")
    user = get_db().get_user(email)
    
    if not user:
        st.error("User not found")
        return
    
    household_id = user.get("household_id")
    
    st.markdown("## 🔔 Notifications & Alerts")
    st.markdown("Configure and test email notifications for bill alerts and optimization tips.")
    
    # Email configuration status
    notif_service = get_notification_service()
    email_status = notif_service.email_backend or "Not configured"
    sms_status = notif_service.sms_backend or "Not configured"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Email Backend", email_status.upper() if isinstance(email_status, str) else "SendGrid" if email_status == "sendgrid" else "SMTP")
    with col2:
        st.metric("SMS Backend", sms_status.upper() if isinstance(sms_status, str) else "Twilio")
    
    # Test notification button
    st.markdown("### Send Test Notification")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_recipient = st.text_input("Test email address", value=email, placeholder="test@example.com")
    
    with col2:
        st.markdown("&nbsp;")  # Alignment
        if st.button("Send Test", type="primary"):
            success, msg = notif_service.send_test_notification(household_id, test_recipient)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    
    # Notification log
    st.markdown("### Notification History")
    log = get_db().get_notification_log(household_id, limit=20)
    
    if log:
        for notif in log:
            status_icon = "✓" if notif.get("status") == "sent" else "✗"
            st.markdown(
                f"{status_icon} **{notif.get('notification_type', 'Unknown').upper()}** to {notif.get('recipient_email')} "
                f"on {notif.get('sent_at', 'Unknown')}"
            )
            if notif.get("status") == "failed":
                st.caption(f"Error: {notif.get('error_message', 'Unknown error')}")
    else:
        st.info("No notifications sent yet.")


def render_chatbot_ui():
    """Render the Q&A Chatbot section as a dedicated tab/feature."""
    if "auth" not in st.session_state or not st.session_state.auth.get("logged_in"):
        return
    
    email = st.session_state.auth.get("email")
    user = get_db().get_user(email)
    
    if not user:
        st.error("User not found")
        return
    
    household_id = user.get("household_id")
    language = st.session_state.get("lang", "en")
    tariff_rate = user.get("tariff_rate", 8.0)
    
    st.markdown("## 💬 Energy Q&A Chatbot")
    st.markdown("Ask questions about your household's energy usage. All answers are based on real data.")
    
    # Initialize chat history
    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history = []
    
    # Display chat history
    if st.session_state.chatbot_history:
        st.markdown("### Conversation History")
        for turn in st.session_state.chatbot_history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                if not turn.get("is_valid"):
                    st.warning("⚠️ Answer validated and may have been corrected for accuracy.")
    
    # Input for new question
    st.markdown("---")
    st.markdown("### Ask a Question")
    
    question = st.text_area(
        "Your question about energy usage:",
        placeholder="e.g., 'Which appliance uses the most power?' or 'What's my predicted monthly cost?'",
        height=100,
        key="chatbot_question_input"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Ask", type="primary"):
            if question.strip():
                with st.spinner("Analyzing your data..."):
                    chatbot = get_chatbot()
                    answer, is_valid, metadata = chatbot.answer_question(
                        household_id=household_id,
                        email=email,
                        question=question,
                        language=language,
                        tariff_rate=tariff_rate
                    )
                    
                    # Add to history
                    st.session_state.chatbot_history.append({
                        "question": question,
                        "answer": answer,
                        "is_valid": is_valid,
                        "metadata": metadata
                    })
                    
                st.rerun()
    
    # Clear history button
    with col1:
        if st.button("Clear Conversation"):
            st.session_state.chatbot_history = []
            st.rerun()


def render_admin_panel():
    """Render an admin panel for testing and debugging (optional)."""
    if "auth" not in st.session_state or not st.session_state.auth.get("logged_in"):
        return
    
    # Only show for non-guest users
    if st.session_state.auth.get("guest"):
        return
    
    with st.expander("🛠️ Admin Panel (Testing)"):
        st.markdown("### Database Status")
        db = get_db()
        email = st.session_state.auth.get("email")
        user = db.get_user(email)
        
        if user:
            st.json({
                "email": user.get("email"),
                "name": user.get("name"),
                "household_id": user.get("household_id"),
                "language": user.get("preferred_language"),
            })
        
        st.markdown("### Manual Notification Trigger (for testing)")
        if st.button("Trigger Bill Alert"):
            notif_service = get_notification_service()
            sent_to = notif_service.send_bill_alert(
                household_id=user.get("household_id"),
                primary_email=email,
                month="2024-09",
                predicted_cost=1250.50,
                language=st.session_state.get("lang", "en")
            )
            st.success(f"Bill alert sent to: {', '.join(sent_to)}")
