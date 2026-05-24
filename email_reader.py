def get_sample_messages():
    return [
        {
            "from": "manager@company.com",
            "subject": "Project Update",
            "body": "Please share the latest project status by EOD today."
        },
        {
            "from": "hr@company.com",
            "subject": "Policy Update",
            "body": "We've updated the leave policy effective next month. Please review it."
        },
        {
            "from": "teammate@company.com",
            "subject": "Weekly Sync",
            "body": "Can we reschedule the weekly sync to Friday?"
        },
        {
            "from": "alerts@bank.com",
            "subject": "Transaction Alert",
            "body": "A transaction of ₹5,000 was made on your card ending with 1234."
        }
    ]


def get_extended_sample_messages():
    base_messages = get_sample_messages()
    additional_messages = [
        {
            "from": "client@fintech.io",
            "subject": "Urgent: Contract Redlines",
            "body": "Please review the updated contract redlines and share legal feedback before 4 PM."
        },
        {
            "from": "security@company.com",
            "subject": "Unusual Login Attempt",
            "body": "We detected an unusual login attempt from a new location. Confirm if this was you."
        },
        {
            "from": "recruiter@startup.ai",
            "subject": "Interview Availability",
            "body": "Can you share your availability this week for a final round product interview?"
        },
        {
            "from": "billing@saasapp.com",
            "subject": "Invoice Overdue Notice",
            "body": "Your monthly subscription invoice is overdue by 7 days. Please complete payment to avoid service interruption."
        },
        {
            "from": "marketing@vendor.com",
            "subject": "Quarterly Webinar Invite",
            "body": "You are invited to our quarterly product webinar next Tuesday. Let us know if you can attend."
        },
        {
            "from": "ops@company.com",
            "subject": "Production Incident Follow-up",
            "body": "Please join the postmortem thread and add your analysis for the outage between 2 AM and 3 AM."
        },
        {
            "from": "support@client.org",
            "subject": "Escalation: Failed Data Export",
            "body": "Our team cannot export reports after the latest update. This is blocking weekly reporting."
        },
        {
            "from": "founder@startup.io",
            "subject": "Product Demo Feedback",
            "body": "Great demo today. Could you send a concise summary and proposed next milestones?"
        }
    ]
    return base_messages + additional_messages
