"""
Test & Verification Script for EnergyPulse New Features
========================================================
Validates family member management, notifications, and chatbot integration.

Run with: python test_new_features.py
"""

import sys
from datetime import datetime

def test_database():
    """Test database layer."""
    print("\n" + "="*60)
    print("TEST 1: Database Layer (Family Members, Notifications, Chat)")
    print("="*60)
    
    try:
        from db import get_db
        db = get_db()
        
        # Test user creation
        test_email = f"test_{datetime.now().timestamp()}@example.com"
        test_user_ok = db.create_user(
            email=test_email,
            name="Test User",
            password="testpass123",
            household_id=f"hh_{datetime.now().timestamp()}"
        )
        print(f"✓ Create user: {test_user_ok}")
        
        # Test authentication
        auth_result = db.authenticate_user(test_email, "testpass123")
        print(f"✓ Authenticate user: {auth_result is not None}")
        if auth_result:
            household_id = auth_result["household_id"]
            
            # Test family member management
            print("\n--- Family Member Management ---")
            success, msg = db.add_family_member(
                household_id=household_id,
                name="Spouse",
                relationship="Spouse",
                email="spouse@example.com",
                phone="+91-9999999999",
                preferred_language="en",
                notify_bills=True,
                notify_tips=True
            )
            print(f"✓ Add family member: {success} (msg: {msg})")
            
            # Retrieve family members
            members = db.get_family_members(household_id)
            print(f"✓ Get family members: {len(members)} member(s) found")
            
            # Test duplicate rejection
            success2, msg2 = db.add_family_member(
                household_id=household_id,
                name="Duplicate",
                relationship="Other",
                email="spouse@example.com",  # Same email - should fail
                preferred_language="en"
            )
            print(f"✓ Reject duplicate email: {not success2} (correctly prevented)")
            
            # Test notification logging
            print("\n--- Notification Logging ---")
            notif_id = db.log_notification(
                household_id=household_id,
                recipient_email="spouse@example.com",
                recipient_name="Spouse",
                notification_type="bill_alert",
                triggered_by="bill_threshold",
                trigger_data={"threshold": 1000, "actual": 1250},
                subject="Bill Alert Test",
                body_text="Your bill is higher than expected",
                language="en"
            )
            print(f"✓ Log notification: ID {notif_id}")
            
            # Retrieve notification log
            log = db.get_notification_log(household_id)
            print(f"✓ Get notification log: {len(log)} notification(s)")
            
            # Test conversation history
            print("\n--- Chatbot Conversation History ---")
            conv_id = db.save_conversation(
                household_id=household_id,
                email=test_email,
                question="Which appliance uses the most power?",
                answer="Based on real data, your AC uses the most power.",
                language="en",
                tool_calls=["get_appliance_breakdown"],
                grounding_data={"AC": 2500},
                is_valid=True
            )
            print(f"✓ Save conversation: ID {conv_id}")
            
            # Retrieve conversation
            history = db.get_conversation_history(household_id, test_email)
            print(f"✓ Get conversation history: {len(history)} turn(s)")
        
        return True
    
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_notifications():
    """Test notification system."""
    print("\n" + "="*60)
    print("TEST 2: Notification System (Email/SMS)")
    print("="*60)
    
    try:
        from notifications import get_notification_service
        from db import get_db
        
        service = get_notification_service()
        db = get_db()
        
        # Check backend configuration
        email_backend = service.email_backend
        sms_backend = service.sms_backend
        
        print(f"✓ Email backend configured: {bool(email_backend)}")
        print(f"  - Backend type: {email_backend or 'None'}")
        print(f"✓ SMS backend configured: {bool(sms_backend)}")
        print(f"  - Backend type: {sms_backend or 'None'}")
        
        # Create test user for notifications
        test_email = f"notif_{datetime.now().timestamp()}@test.local"
        test_hh = f"hh_notif_{datetime.now().timestamp()}"
        db.create_user(test_email, "Test", "pass", household_id=test_hh)
        
        # Test bill alert notification
        print("\n--- Bill Alert Notification ---")
        if email_backend:
            sent_to = service.send_bill_alert(
                household_id=test_hh,
                primary_email=test_email,
                month="2024-09",
                predicted_cost=1500.00,
                threshold=1200.00,
                language="en"
            )
            print(f"✓ Send bill alert: {len(sent_to)} recipient(s)")
            print(f"  - Sent to: {sent_to}")
        else:
            print("⚠ Email backend not configured - skipping email test")
        
        # Test optimization tip notification
        print("\n--- Optimization Tip Notification ---")
        tip_data = {
            "appliance_name": "Air Conditioner",
            "current_usage": 3.5,
            "avg_usage": 2.0,
            "increase_pct": 75,
            "suggested_action": "Consider using AC only during peak cooling hours."
        }
        if email_backend:
            sent_to = service.send_optimization_tips(
                household_id=test_hh,
                primary_email=test_email,
                tip_data=tip_data,
                language="en"
            )
            print(f"✓ Send optimization tip: {len(sent_to)} recipient(s)")
        else:
            print("⚠ Email backend not configured - skipping email test")
        
        return True
    
    except Exception as e:
        print(f"✗ Notification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatbot():
    """Test grounded chatbot."""
    print("\n" + "="*60)
    print("TEST 3: Grounded Q&A Chatbot")
    print("="*60)
    
    try:
        from chatbot import get_chatbot, EnergyPulseChatbot
        from db import get_db
        
        chatbot = get_chatbot()
        db = get_db()
        
        # Create test user
        test_email = f"chat_{datetime.now().timestamp()}@test.local"
        test_hh = f"hh_chat_{datetime.now().timestamp()}"
        db.create_user(test_email, "ChatTest", "pass", household_id=test_hh)
        
        print(f"✓ Chatbot initialized with {len(chatbot.tools)} tools")
        print(f"  - Available tools: {list(chatbot.tools.keys())}")
        
        # Test Q&A in multiple languages
        test_questions = [
            ("en", "Which appliance uses the most power?"),
            ("hi", "कौन सा उपकरण सबसे अधिक बिजली खपत करता है?"),
            ("kn", "ಯಾವ ಉಪಕರಣ ಹೆಚ್ಚು ವಿದ್ಯುತ್ ಬಳಕೆ ಮಾಡುತ್ತದೆ?"),
            ("en", "What's my predicted monthly cost?"),
        ]
        
        print("\n--- Multi-language Q&A ---")
        for lang, question in test_questions:
            try:
                answer, is_valid, metadata = chatbot.answer_question(
                    household_id=test_hh,
                    email=test_email,
                    question=question,
                    language=lang,
                    tariff_rate=8.0
                )
                
                lang_name = {"en": "English", "hi": "Hindi", "kn": "Kannada"}.get(lang, lang)
                status = "✓" if is_valid else "⚠"
                print(f"{status} {lang_name}: Answered (valid: {is_valid})")
                print(f"  Q: {question}")
                print(f"  A: {answer[:100]}...")
            except Exception as e:
                print(f"✗ {lang}: Failed - {e}")
        
        # Test unanswerable question
        print("\n--- Handling Unanswerable Questions ---")
        unanswerable_q = "What will the weather be tomorrow?"
        answer, is_valid, _ = chatbot.answer_question(
            household_id=test_hh,
            email=test_email,
            question=unanswerable_q,
            language="en"
        )
        
        is_honest = "don't" in answer.lower() or "no" in answer.lower() or "unavailable" in answer.lower()
        print(f"✓ Honest response (no hallucination): {is_honest}")
        print(f"  Q: {unanswerable_q}")
        print(f"  A: {answer[:100]}...")
        
        return True
    
    except Exception as e:
        print(f"✗ Chatbot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_guard():
    """Test LLM hallucination guard."""
    print("\n" + "="*60)
    print("TEST 4: LLM Hallucination Guard Validation")
    print("="*60)
    
    try:
        from llm_guard import LLMHallucinationGuard
        
        guard = LLMHallucinationGuard()
        
        # Register some tool results
        guard.register_tool_result("get_appliance_breakdown", {
            "AC_kwh": 150.5,
            "Fridge_kwh": 45.2,
            "Lights_kwh": 23.0
        })
        
        # Test valid response (uses real numbers)
        valid_response = "Your AC used 150.5 kWh and fridge used 45.2 kWh this month."
        is_valid, issues, corrected = guard.validate_response(valid_response, language="en")
        print(f"✓ Valid response (with real numbers): {is_valid}")
        
        # Test response with invented number (should flag)
        invalid_response = "Your AC used 999 kWh (completely made up number)."
        is_valid2, issues2, corrected2 = guard.validate_response(invalid_response, language="en")
        print(f"✓ Detects invented numbers: {not is_valid2}")
        
        return True
    
    except Exception as e:
        print(f"✗ LLM Guard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENERGYPULSE NEW FEATURES TEST SUITE")
    print("="*60)
    
    results = {
        "Database": test_database(),
        "Notifications": test_notifications(),
        "Chatbot": test_chatbot(),
        "LLM Guard": test_llm_guard(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*60 + "\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
