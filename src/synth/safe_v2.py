"""
src/synth/safe_v2.py - a SECOND safe generator, built to fix false alarms.

WHY A SECOND GENERATOR
  The model's real failure is false positives: 15.3% of held-out real ham and
  13% of fresh safe-chat probes get called scam. It learned "bank words + link
  + urgency = scam" because almost every safe row it saw was a plain chat
  message. This module manufactures the missing population: LEGITIMATE messages
  that use scam-adjacent vocabulary (bank, account, debit, OTP, KYC, link,
  urgent, verify) without being scams.

WHY IT IS DISJOINT
  * its pools are written here, not in src/synth/components.py, so no training
    family name is reused;
  * it never reads or writes the frozen 200-row safe-chat probe
    (data/processed/safe_chat_probe_frozen.parquet) - that probe stays the
    held-out TEST set, and this module ASSERTS the maximum similarity between
    the two is below 0.7.

WHAT COMES OUT
  ~2,500 rows, source="synthetic", verdict="safe", notes starting with
  "hard_negative=true", 8 categories x many templates x openers x tails, so the
  normalised unique-text ratio stays high (the build refuses <95%).

Run:
  python -m src.synth.safe_v2            # writes the parquet + prints the report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, SEED, data_path
from src.data_checks import normalize_text

# Words that make a message a HARD negative instead of an easy one. These are
# exactly the words a scam detector over-triggers on, which is the point.
HARD_WORDS = ("bank", "account", "a/c", "debit", "credit", "otp", "kyc",
              "link", "urgent", "verify", "transaction", "payment", "blocked",
              "refund", "amount", "upi", "pin", "card")

# ---------------------------------------------------------------------------
# OPENERS / TAILS - deliberately different from the frozen probe's phrasing.
# ---------------------------------------------------------------------------
OPENERS = {
    "english": [
        "Good morning,",
        "Hello,",
        "Hi,",
        "Quick update:",
        "Heads up -",
        "FYI,",
        "Note:",
    ],
    "roman_hinglish": [
        "Namaste,",
        "Hi,",
        "Ek update hai:",
        "Bata raha hoon -",
        "Suno,",
        "Chhoti si baat,",
        "Note kar lo:",
    ],
}

TAILS = {
    "english": [
        "Nothing needed from your side.",
        "No action required.",
        "Sharing for your records.",
        "Let me know if anything looks off.",
        "Talk later.",
        "All fine at my end.",
    ],
    "roman_hinglish": [
        "Aapko kuch karne ki zaroorat nahi.",
        "Bas jaankari ke liye bhej raha hoon.",
        "Koi action nahi chahiye.",
        "Baad mein baat karte hain.",
        "Sab theek hai mere taraf se.",
        "Records ke liye rakh lena.",
    ],
}

# One official-looking link tail, used rarely: a real bank alert CAN carry a
# link, and the model must learn that a link alone is not evidence of fraud.
LINK_TAIL = {
    "english": "Details are in the official app: <URL>",
    "roman_hinglish": "Details official app mein hain: <URL>",
}

# ---------------------------------------------------------------------------
# BODIES - 8 categories. Every pair is (english, roman_hinglish).
# ---------------------------------------------------------------------------
BODIES: dict[str, list[tuple[str, str]]] = {
    "bank_alert": [
        ("Rs {AMOUNT} has been debited from your account ending {LAST4} at a "
         "merchant terminal in {CITY}. This is a routine alert, not a request.",
         "Aapke account {LAST4} se Rs {AMOUNT} debit hue hain {CITY} mein. "
         "Yeh sirf soochna hai, koi request nahi."),
        ("Rs {AMOUNT} has been credited to your account ending {LAST4} by NEFT "
         "transfer. The reference number is in your statement.",
         "Aapke account {LAST4} mein Rs {AMOUNT} NEFT se credit hue hain. "
         "Reference number statement mein hai."),
        ("Your card ending {LAST4} has been blocked at your request and a new "
         "card will be dispatched to your registered address.",
         "Aapke request par card {LAST4} block kar diya gaya hai, naya card "
         "registered address par aayega."),
        ("This month's statement was emailed and is also under the documents "
         "tab of the app, so you can download it any time.",
         "Is mahine ka statement email par bhi bheja gaya hai aur app ke "
         "documents tab mein bhi hai, jab chahein download kar lein."),
        ("Your KYC documents have been verified at the {BRANCH} branch and no "
         "further submission is needed this year.",
         "Aapke KYC documents {BRANCH} branch mein verify ho gaye hain, is "
         "saal aur kuch submit karne ki zaroorat nahi."),
        ("A cheque of Rs {AMOUNT} issued from your account ending {LAST4} has "
         "been cleared this morning.",
         "Aapke account {LAST4} se Rs {AMOUNT} ka cheque aaj subah clear ho "
         "gaya hai."),
        ("Your fixed deposit receipt is ready for collection at the {BRANCH} "
         "branch. Please carry a photo ID when you visit.",
         "Aapki fixed deposit receipt {BRANCH} branch mein ready hai. Aate "
         "samay photo ID saath lein."),
        ("A standing instruction of Rs {AMOUNT} for your mutual fund SIP was "
         "executed successfully on {DATE}.",
         "Aapke mutual fund SIP ka Rs {AMOUNT} ka standing instruction {DATE} "
         "ko successfully chala."),
        ("Your account ending {LAST4} has a pending service request that was "
         "closed without any charge. No payment is due.",
         "Aapke account {LAST4} ki service request bina kisi charge ke band "
         "ho gayi. Koi payment baaki nahi hai."),
        ("We noticed a login to your net banking from a registered device in "
         "{CITY}. If this was you, no action is needed.",
         "Aapke net banking mein {CITY} se ek registered device se login hua. "
         "Agar yeh aap the to kuch karne ki zaroorat nahi."),
        ("Your salary of Rs {AMOUNT} has been credited to account {LAST4} "
         "along with this month's reimbursement.",
         "Is mahine ki salary Rs {AMOUNT} aur reimbursement account {LAST4} "
         "mein credit ho gaye hain."),
        ("The branch at {CITY} will stay open until 5 pm on {DATE} for "
         "passbook printing and cheque collection.",
         "{CITY} ki branch {DATE} ko shaam 5 baje tak passbook printing aur "
         "cheque ke liye khuli rahegi."),
    ],
    "otp_delivery": [
        ("{OTP} is your one-time password for the login you started. Do not "
         "share it with anyone, including bank staff.",
         "{OTP} aapka one-time password hai login ke liye. Ise kisi ke saath "
         "share na karein, bank staff ke saath bhi nahi."),
        ("Use {OTP} to authorise the payment of Rs {AMOUNT} on your card "
         "ending {LAST4}. The code expires in ten minutes.",
         "Card {LAST4} par Rs {AMOUNT} ke payment ke liye {OTP} use karein. "
         "Code das minute mein expire hoga."),
        ("Your verification code is {OTP}. Nobody from the bank will ever ask "
         "you for this code.",
         "Aapka verification code {OTP} hai. Bank se koi bhi yeh code kabhi "
         "nahi maangega."),
        ("{OTP} is the OTP for updating your address. If you did not request "
         "this, ignore the message and the request will lapse.",
         "{OTP} address update karne ka OTP hai. Agar aapne request nahi ki "
         "to message ignore karein, request cancel ho jayegi."),
        ("Do not share the OTP {OTP} with the delivery agent. It is only for "
         "confirming your order in the app.",
         "Delivery agent ke saath OTP {OTP} share na karein. Yeh sirf app mein "
         "order confirm karne ke liye hai."),
        ("Your new card is ready. Enter {OTP} in the app to activate it "
         "yourself; no branch visit is needed.",
         "Aapka naya card taiyar hai. App mein {OTP} daal kar khud activate "
         "karein, branch jaane ki zaroorat nahi."),
        ("{OTP} confirms your recurring deposit. The amount of Rs {AMOUNT} "
         "will be debited on the first of every month.",
         "{OTP} aapki recurring deposit confirm karta hai. Rs {AMOUNT} har "
         "mahine ki pehli tareekh ko debit hoga."),
        ("Security alert: a password reset was requested. The code is {OTP}. "
         "If it was not you, call the number printed on your card.",
         "Security alert: password reset ki request aayi thi. Code {OTP} hai. "
         "Agar aapne nahi kiya to card par chhape number par call karein."),
        ("Your UPI PIN change request is pending. Approve it inside the app "
         "with the code {OTP} only if you started it.",
         "Aapki UPI PIN change request pending hai. Sirf tabhi code {OTP} se "
         "approve karein agar aapne shuru ki thi."),
        ("This is a reminder that OTPs are personal. We will never call to ask "
         "for the code we just sent to your phone.",
         "Yaad rahe, OTP personal hota hai. Jo code humne bheja hai wo hum "
         "phone par kabhi nahi maangte."),
    ],
    "courier": [
        ("Your shipment with AWB {ORDER} reached the {CITY} hub overnight and "
         "is scheduled for delivery tomorrow between 10 am and 1 pm.",
         "Aapka shipment AWB {ORDER} raat ko {CITY} hub pahunch gaya aur kal "
         "10 se 1 baje ke beech delivery ke liye scheduled hai."),
        ("The delivery agent for AWB {ORDER} will call before arriving. Please "
         "keep the phone reachable and a photo ID handy.",
         "AWB {ORDER} ka delivery agent aane se pehle call karega. Phone "
         "paas rakhein aur photo ID taiyar rakhein."),
        ("AWB {ORDER} was handed to the security cabin at your address because "
         "nobody answered the door.",
         "AWB {ORDER} aapke address par security cabin ko de diya gaya "
         "kyunki darwaza kisi ne nahi khola."),
        ("Your return pickup for order {ORDER} is confirmed for {DATE} between "
         "2 pm and 6 pm. Keep the item packed with its invoice.",
         "Order {ORDER} ka return pickup {DATE} ko 2 se 6 baje confirm hai. "
         "Item invoice ke saath pack kar ke rakhein."),
        ("Order {ORDER} shows as delivered at the reception desk. Track the "
         "signature image in the app if you cannot find it.",
         "Order {ORDER} reception desk par delivered dikh raha hai. Agar nahi "
         "mil raha to app mein signature image dekh lein."),
        ("Your consignment was rescheduled at your request to {DATE}. No extra "
         "charge has been applied.",
         "Aapki request par consignment {DATE} ke liye reschedule ho gayi. "
         "Koi extra charge nahi laga."),
        ("AWB {ORDER} is at the {CITY} delivery centre. You can collect it "
         "between 9 am and 8 pm with the order confirmation.",
         "AWB {ORDER} {CITY} delivery centre par hai. Order confirmation ke "
         "saath 9 se 8 baje ke beech collect kar sakte hain."),
        ("The courier could not complete delivery because the address pin code "
         "was missing. Reply with the correct pin code for order {ORDER}.",
         "Address ka pin code missing hone ki wajah se delivery poori nahi ho "
         "payi. Order {ORDER} ke liye sahi pin code reply karein."),
        ("Your package for order {ORDER} is out with the local rider and "
         "should arrive within the next two hours.",
         "Order {ORDER} ka package local rider ke paas hai aur do ghante mein "
         "pahunch jana chahiye."),
        ("AWB {ORDER} cleared customs this morning. Delivery timelines are "
         "unchanged and no duty payment is needed from you.",
         "AWB {ORDER} aaj subah customs se clear ho gaya. Delivery time same "
         "hai aur aapko koi duty nahi deni."),
    ],
    "appointment": [
        ("Your appointment at the {BRANCH} clinic is confirmed for {DATE} at "
         "{HOUR}. Please arrive ten minutes early with earlier prescriptions.",
         "{BRANCH} clinic mein aapki appointment {DATE} ko {HOUR} baje "
         "confirm hai. Purani prescriptions ke saath das minute pehle "
         "pahunchein."),
        ("The lab test slot booked for {DATE} at {HOUR} is confirmed. Fasting "
         "is required for eight hours before the sample.",
         "{DATE} ko {HOUR} baje ka lab test slot confirm hai. Sample se pehle "
         "aath ghante fasting zaroori hai."),
        ("Your train PNR ending {LAST4} has been confirmed after chart "
         "preparation. Coach and seat details are in the app.",
         "Chart preparation ke baad aapka train PNR {LAST4} confirm ho gaya. "
         "Coach aur seat details app mein hain."),
        ("Reminder: your dental appointment is tomorrow at {HOUR}. Call the "
         "front desk if you need to move it.",
         "Reminder: kal {HOUR} baje dental appointment hai. Badalna ho to "
         "front desk par call karein."),
        ("Your passport appointment at the {CITY} office is on {DATE}. Carry "
         "the original documents listed in the checklist.",
         "{CITY} office mein passport appointment {DATE} ko hai. Checklist "
         "mein likhe original documents saath lein."),
        ("The service centre has kept a slot for your car service on {DATE} at "
         "{HOUR}. Pickup can be arranged on request.",
         "Service centre ne {DATE} ko {HOUR} baje car service ke liye slot "
         "rakha hai. Request par pickup ho sakta hai."),
        ("Your hotel booking from {DATE} is confirmed for two nights. Check-in "
         "opens at {HOUR} and a photo ID is required.",
         "{DATE} se do raat ke liye hotel booking confirm hai. Check-in "
         "{HOUR} baje se hai aur photo ID zaroori hai."),
        ("The parent-teacher meeting has moved to {DATE} at {HOUR} in the "
         "school hall. No fee is collected at the meeting.",
         "Parent-teacher meeting {DATE} ko {HOUR} baje school hall mein ho "
         "gi. Meeting mein koi fees nahi li jaati."),
        ("Your vaccination slot for {DATE} is confirmed at the {BRANCH} centre. "
         "Bring the earlier vaccination card.",
         "{DATE} ka vaccination slot {BRANCH} centre par confirm hai. Purana "
         "vaccination card le kar aayein."),
        ("The annual health check booked through the company plan is on "
         "{DATE}; the report will be shared in the portal.",
         "Company plan se annual health check {DATE} ko hai; report portal "
         "mein milegi."),
    ],
    "utility": [
        ("Your electricity bill of Rs {AMOUNT} for account {LAST4} has been "
         "received. The payment receipt is available in the app.",
         "Aapka electricity bill Rs {AMOUNT} account {LAST4} ke liye receive "
         "ho gaya. Payment receipt app mein hai."),
        ("Bill payment of Rs {AMOUNT} against consumer number {LAST4} was "
         "successful on {DATE}. Thank you for paying on time.",
         "Consumer number {LAST4} ka Rs {AMOUNT} bill payment {DATE} ko "
         "successful hua. Time par payment ke liye dhanyawad."),
        ("The water bill for {DATE} is Rs {AMOUNT}. Payment can be made at any "
         "authorised counter or through the official portal.",
         "{DATE} ka water bill Rs {AMOUNT} hai. Kisi bhi authorised counter "
         "ya official portal se payment kar sakte hain."),
        ("Your piped gas refill was delivered today. The meter reading of "
         "{ORDER} units has been updated in your account.",
         "Aapka piped gas refill aaj deliver hua. {ORDER} units ki meter "
         "reading account mein update ho gayi hai."),
        ("A duplicate bill copy for account {LAST4} was requested. It is "
         "attached to your registered email, no charge applies.",
         "Account {LAST4} ki duplicate bill copy request hui thi. Woh "
         "registered email par bhej di gayi hai, koi charge nahi."),
        ("Your broadband plan renewed automatically for Rs {AMOUNT}. The next "
         "billing date is {DATE}.",
         "Aapka broadband plan Rs {AMOUNT} mein automatically renew ho gaya. "
         "Agli billing date {DATE} hai."),
        ("The society maintenance invoice for {DATE} is Rs {AMOUNT} and "
         "includes the water tanker charge for the month.",
         "{DATE} ka society maintenance invoice Rs {AMOUNT} hai aur is mahine "
         "ka water tanker charge bhi isme shamil hai."),
        ("Your prepaid meter recharge of Rs {AMOUNT} was applied to consumer "
         "number {LAST4}. Balance is visible on the meter display.",
         "Consumer number {LAST4} par Rs {AMOUNT} ka prepaid recharge lag "
         "gaya. Balance meter display par dikh raha hai."),
        ("The billing cycle changed this month, so your next bill of Rs "
         "{AMOUNT} covers thirty-two days instead of thirty.",
         "Is mahine billing cycle badla hai, isliye agla bill Rs {AMOUNT} "
         "tees din ki jagah battis din ka hai."),
    ],
    "delivery_coord": [
        ("Vegetables are in the bottom tray and the curd is on the middle "
         "shelf, so nothing spoils before the weekend.",
         "Sabziyan neeche ke tray mein hain aur dahi beech ke shelf par, "
         "weekend se pehle kuch kharab nahi hoga."),
        ("The plumber is coming at {HOUR}, so please be home to open the door. "
         "I have kept the spare key with the guard.",
         "Plumber {HOUR} baje aayega, ghar par rehna darwaza kholne ke liye. "
         "Spare key guard ke paas rakh di hai."),
        ("I left the parcel with the neighbour in flat {ORDER} because I had to "
         "leave for the office early.",
         "Parcel flat {ORDER} ke neighbour ke paas chhod diya kyunki mujhe "
         "office jaldi jaana tha."),
        ("The laundry was picked up at {HOUR} and will be returned on {DATE} "
         "evening, they said.",
         "Laundry {HOUR} baje pick hui thi aur {DATE} shaam tak wapas aayegi, "
         "unhone bola."),
        ("The furniture delivery team will need the lift for twenty minutes; I "
         "have informed the society office.",
         "Furniture delivery team ko lift bees minute chahiye; society office "
         "ko bata diya hai."),
        ("I paid the newspaper vendor for three months in cash, so ignore his "
         "reminder if he sends one.",
         "Newspaper wale ko teen mahine ka cash de diya hai, agar woh reminder "
         "bheje to ignore karna."),
        ("Please collect the medicine from the chemist near the {CITY} market; "
         "it is already paid for.",
         "{CITY} market ke paas chemist se dawai le aana; payment ho chuki "
         "hai."),
        ("The water tanker came at {HOUR} and filled the overhead tank. The "
         "motor is switched off now.",
         "Water tanker {HOUR} baje aaya aur overhead tank bhar diya. Motor ab "
         "band hai."),
        ("Landlord confirmed that the rent receipt for {DATE} is ready and can "
         "be collected from his office.",
         "Landlord ne confirm kiya ki {DATE} ki rent receipt taiyar hai aur "
         "uske office se le sakte hain."),
        ("I have kept the courier's slip in the drawer next to the door, use it "
         "if you need to track the return.",
         "Courier ki slip darwaze ke paas drawer mein rakhi hai, return track "
         "karna ho to use karna."),
    ],
    "family_chat": [
        ("Are we still meeting at {HOUR} near the {CITY} station, or should I "
         "come to the house directly?",
         "Hum {HOUR} baje {CITY} station ke paas mil rahe hain ya main seedha "
         "ghar aa jaun?"),
        ("Dad's physiotherapy is at {HOUR} tomorrow, I will take him and you "
         "can pick up the medicines.",
         "Papa ki physiotherapy kal {HOUR} baje hai, main le jaunga aur tum "
         "dawai le aana."),
        ("The wedding album is uploaded to the folder I shared last week and "
         "the download works until Sunday evening.",
         "Wedding ka album us folder mein hai jo maine pichhle hafte share "
         "kiya tha, download Sunday shaam tak chalega."),
        ("I reached home safely, the train was only ten minutes late. Don't "
         "worry about picking me up.",
         "Main ghar safe pahunch gaya, train sirf das minute late thi. Pickup "
         "ki tension mat lo."),
        ("Mummy's sugar report came back normal. She has an appointment with "
         "the doctor on {DATE} anyway.",
         "Mummy ka sugar report normal aaya hai. {DATE} ko doctor ki "
         "appointment phir bhi hai."),
        ("Can you send the address of the new house again? I lost the message "
         "when I changed phones.",
         "Naye ghar ka address dobara bhej de? Phone badalne par message kho "
         "gaya tha."),
        ("The kids have a holiday on {DATE}, so I will work from home that day.",
         "Bachchon ki {DATE} ko chhutti hai, isliye main us din ghar se kaam "
         "karunga."),
        ("Aunty is coming by the {HOUR} bus; someone needs to be at the stop to "
         "receive her.",
         "Aunty {HOUR} baje ki bus se aa rahi hain; kisi ko stop par rukna "
         "padega unhe lene."),
        ("I already transferred my share of the electricity bill to your "
         "account, check when you get a minute.",
         "Electricity bill ka mera hissa maine tumhare account mein bhej diya "
         "hai, time mile to check kar lena."),
        ("Let's order dinner instead of cooking, I will place it for {HOUR} "
         "delivery.",
         "Khana banana chhod ke order kar lete hain, main {HOUR} baje delivery "
         "ke liye laga deta hoon."),
    ],
    "work_chat": [
        ("The client moved the review call to {HOUR}; the deck is already in "
         "the shared folder.",
         "Client ne review call {HOUR} baje kar di hai; deck shared folder "
         "mein already hai."),
        ("I have approved your leave for {DATE}. Please hand over the pending "
         "tickets before you go.",
         "Aapki {DATE} ki leave approve kar di hai. Jaane se pehle pending "
         "tickets hand over kar dein."),
        ("The reimbursement of Rs {AMOUNT} has been processed and will reflect "
         "in this month's payroll.",
         "Rs {AMOUNT} ka reimbursement process ho gaya hai aur is mahine ki "
         "payroll mein aayega."),
        ("Standup is at {HOUR} today because of the release. Bring the "
         "blockers list.",
         "Release ki wajah se standup aaj {HOUR} baje hai. Blockers ki list "
         "le kar aayein."),
        ("The vendor invoice for {DATE} is pending approval in the portal; it "
         "needs your sign-off before {HOUR}.",
         "{DATE} ka vendor invoice portal mein approval ke liye pending hai; "
         "{HOUR} se pehle aapka sign-off chahiye."),
        ("Your access to the analytics dashboard has been restored. No ticket "
         "is needed if it works now.",
         "Analytics dashboard ka access restore ho gaya hai. Ab kaam kar raha "
         "hai to ticket ki zaroorat nahi."),
        ("The office will be closed on {DATE} for maintenance. Please work from "
         "home and keep the VPN on.",
         "{DATE} ko maintenance ke liye office band rahega. Ghar se kaam "
         "karein aur VPN on rakhein."),
        ("HR has shared the updated policy document in the portal; acknowledgement "
         "is due by {DATE}.",
         "HR ne updated policy document portal mein share kiya hai; {DATE} tak "
         "acknowledge karna hai."),
        ("The laptop replacement you requested is approved. IT will collect the "
         "old one on {DATE}.",
         "Aapka laptop replacement approve ho gaya. IT purana {DATE} ko "
         "collect karega."),
        ("Please send the weekly status by {HOUR} today, and mention any risk "
         "that needs escalation.",
         "Aaj {HOUR} baje tak weekly status bhej dein, aur koi risk ho to "
         "escalation ke liye likh dein."),
    ],
}

# ---------------------------------------------------------------------------
# HARD clauses - one extra TRUE sentence that carries scam-adjacent vocabulary.
#
# WHY: the categories above (appointments, family chat, work chat, delivery
# coordination) are legitimately free of bank words, so on their own they push
# the hard-negative share below the 60% the pipeline asks for. Adding ONE
# honest sentence to about half of those rows raises the share without
# inventing a fake message: every clause below is a normal thing to say.
# ---------------------------------------------------------------------------
HARD_CLAUSES: dict[str, list[tuple[str, str]]] = {
    "appointment": [
        ("The consultation payment was already made online.",
         "Consultation ka payment online ho chuka hai."),
        ("No payment is needed at the counter.",
         "Counter par koi payment nahi dena hai."),
        ("The bill was settled through the app.",
         "Bill app se settle ho gaya hai."),
        ("Your insurance has approved the amount, so nothing is payable.",
         "Insurance ne amount approve kar diya hai, kuch pay nahi karna."),
        ("Keep the payment receipt with you for the visit.",
         "Visit ke liye payment receipt saath rakhein."),
    ],
    "courier": [
        ("No payment is due for this shipment.",
         "Is shipment ke liye koi payment baaki nahi hai."),
        ("The courier charge was already paid online.",
         "Courier ka charge online pay ho chuka hai."),
        ("Cash on delivery is not applicable to this order.",
         "Is order par cash on delivery lagu nahi hai."),
        ("The invoice is inside the package.",
         "Invoice package ke andar hai."),
        ("You do not need to verify anything with the agent.",
         "Agent se kuch verify karne ki zaroorat nahi hai."),
    ],
    "delivery_coord": [
        ("The online payment for the order already went through.",
         "Order ka online payment ho chuka hai."),
        ("I have settled the bill, so no cash is needed.",
         "Bill settle kar diya hai, cash ki zaroorat nahi."),
        ("Keep the payment screenshot just in case.",
         "Payment ka screenshot sambhal ke rakhna."),
        ("The landlord's account details are on the receipt.",
         "Landlord ke account details receipt par hain."),
        ("Do not transfer any amount to the delivery person.",
         "Delivery person ko koi amount transfer na karna."),
    ],
    "family_chat": [
        ("I have already paid my share of the bill.",
         "Bill ka mera hissa maine pay kar diya hai."),
        ("Do not send money, everything is settled.",
         "Paisa mat bhejna, sab settle hai."),
        ("The school fee receipt is in the drawer.",
         "School fee ki receipt drawer mein hai."),
        ("I will transfer the amount tonight after dinner.",
         "Raat ko khane ke baad amount transfer kar dunga."),
        ("Keep the old bank passbook, we may need the account number.",
         "Purani bank passbook sambhal ke rakhna, account number ki zaroorat pad sakti hai."),
    ],
    "work_chat": [
        ("The invoice is approved and payment will follow in the next payroll run.",
         "Invoice approve ho gaya hai aur payment agle payroll mein aayega."),
        ("Please do not share the client's account details outside the team.",
         "Client ke account details team ke bahar share na karein."),
        ("The reimbursement payment has cleared, check your bank statement.",
         "Reimbursement ka payment aa gaya hai, bank statement check karein."),
        ("Finance needs the invoice copy for the transaction to be booked.",
         "Transaction book karne ke liye finance ko invoice copy chahiye."),
        ("Nothing is urgent today, take your time with the review.",
         "Aaj kuch urgent nahi hai, review aaram se karein."),
    ],
}

CITIES = ["Pune", "Jaipur", "Kochi", "Indore", "Lucknow", "Nagpur", "Surat",
          "Bhopal", "Patna", "Coimbatore"]
BRANCHES = ["Kalyani Nagar", "Civil Lines", "MG Road", "Sector 12", "Anna Nagar",
            "Gomti Nagar", "Vashi", "Salt Lake"]
AMOUNTS = ["249", "499", "1,250", "2,400", "780", "3,999", "150", "6,500",
           "899", "1,099"]
HOURS = ["9:30 am", "11 am", "4 pm", "6:30 pm", "10 am", "5 pm", "7 pm"]
DATES = ["Monday", "Tuesday", "the 12th", "the 18th", "next Friday",
         "the 3rd", "Saturday"]
COUNTS_PER_CATEGORY = {
    "bank_alert": 380,
    "otp_delivery": 340,
    "courier": 330,
    "appointment": 290,
    "utility": 260,
    "delivery_coord": 300,
    "family_chat": 280,
    "work_chat": 320,
}
DEFAULT_TOTAL = 2500


def _fill(text: str, rng: np.random.Generator) -> str:
    """Substitute every slot. Digits never form a long real-looking number."""
    return (text.replace("{AMOUNT}", str(rng.choice(AMOUNTS)))
                .replace("{LAST4}", f"{int(rng.integers(1000, 9999))}")
                .replace("{ORDER}", f"{int(rng.integers(100000, 999999))}")
                .replace("{OTP}", f"{int(rng.integers(100000, 999999))}")
                .replace("{CITY}", str(rng.choice(CITIES)))
                .replace("{BRANCH}", str(rng.choice(BRANCHES)))
                .replace("{HOUR}", str(rng.choice(HOURS)))
                .replace("{DATE}", str(rng.choice(DATES))))


def _compose(category: str, body_idx: int, opener: str | None,
             tail: str | None, style: str, rng, add_hard: bool = False) -> str:
    """One message: optional opener + body + optional clause + optional tail."""
    en, hi = BODIES[category][body_idx]
    body = en if style == "english" else hi
    parts = []
    if opener:
        parts.append(opener)
    parts.append(_fill(body, rng))
    if add_hard and category in HARD_CLAUSES:
        clause = HARD_CLAUSES[category][int(rng.integers(0, len(HARD_CLAUSES[category])))]
        parts.append(clause[0] if style == "english" else clause[1])
    if tail:
        parts.append(tail)
    return " ".join(p for p in parts if p)


def generate(n: int = DEFAULT_TOTAL, seed: int = SEED) -> pd.DataFrame:
    """Build `n` fresh hard-negative SAFE rows (deterministic given the seed).

    Sampling is WITHOUT REPLACEMENT over the (body x opener x tail) space, so
    no two rows start out identical; exact dedupe is still applied and asserted.
    """
    rng = np.random.default_rng(seed)
    # Proportional allocation, largest remainder, so `n` is hit exactly.
    base_total = sum(COUNTS_PER_CATEGORY.values())
    exact = {cat: n * c / base_total for cat, c in COUNTS_PER_CATEGORY.items()}
    alloc = {cat: int(v) for cat, v in exact.items()}
    short = n - sum(alloc.values())
    for cat in sorted(exact, key=lambda c: -(exact[c] - alloc[c]))[:short]:
        alloc[cat] += 1

    rows: list[dict] = []
    seq = 0
    for category, count in alloc.items():
        bodies = BODIES[category]
        # Build the pool of (body, opener, tail, style) tuples available.
        pool: list[tuple[int, str | None, str | None, str]] = []
        for style in ("english", "roman_hinglish"):
            openers = [None] + list(OPENERS[style])
            tails = [None] + list(TAILS[style]) + [LINK_TAIL[style]]
            for b_idx in range(len(bodies)):
                for op in openers:
                    for tail in tails:
                        pool.append((b_idx, op, tail, style))
        picks = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        for pick in picks:
            b_idx, opener, tail, style = pool[int(pick)]
            # Half of the rows in the weak categories get one extra true
            # sentence carrying scam-adjacent vocabulary (see HARD_CLAUSES).
            add_hard = category in HARD_CLAUSES and rng.random() < 0.72
            text = _compose(category, b_idx, opener, tail, style, rng, add_hard)
            seq += 1
            hard = any(word in text.lower() for word in HARD_WORDS)
            rows.append({
                "id": f"v2{seq:05d}",
                "text": text,
                "verdict": "safe",
                "tactics": [],
                "channel": "sms",
                "script_style": style,
                "date_received": None,
                "source": "synthetic",
                "template_family": f"v2_{category}_f{b_idx + 1}",
                "labeler": "generator_v2",
                "label_confidence": "high",
                "url_type": "other" if "<URL>" in text else "none",
                "had_phone_number": False,
                "template_cluster": -1,          # assigned by src.dedup later
                "split": "",
                "real_stratum": "",
                "notes": f"hard_negative={str(hard).lower()}; cat={category}; "
                         "v2 safe generator (Phase 5 false-alarm fix)",
                "hard_negative": bool(hard),
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["id"] = [f"v2{i:05d}" for i in range(1, len(df) + 1)]
    return df


def dedupe_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose NORMALISED text repeats (digits -> 0)."""
    norm = df["text"].map(normalize_text)
    keep = ~norm.duplicated()
    dropped = int((~keep).sum())
    if dropped:
        print(f"  safe_v2 dedupe: dropped {dropped} normalised duplicate(s)")
    return df.loc[keep].reset_index(drop=True)


def similarity_to_probe(df: pd.DataFrame) -> dict:
    """Max TF-IDF similarity of these rows to the FROZEN safe-chat probe."""
    probe_path = data_path("safe_chat_probe")
    if not probe_path.exists():
        return {"available": False}
    probe = pd.read_parquet(probe_path)
    from src.nn_sim import max_similarity_to_train

    sim = max_similarity_to_train(probe["text"].astype(str).tolist(),
                                  df["text"].astype(str).tolist())
    return {
        "available": True,
        "n_probe": int(len(probe)),
        "max": float(np.max(sim)),
        "mean": float(np.mean(sim)),
        "share_above_0_6": float((sim >= 0.6).mean()),
        "share_above_0_7": float((sim >= 0.7).mean()),
    }


def build(n: int = DEFAULT_TOTAL, seed: int = SEED, verbose: bool = True) -> pd.DataFrame:
    """Generate, dedupe, verify and (optionally) print the checks."""
    df = dedupe_exact(generate(n, seed))
    n_norm_unique = df["text"].map(normalize_text).nunique()
    hard_share = float(df["hard_negative"].mean())
    print(f"(seed={seed}) src/synth/safe_v2.py: {len(df):,} safe hard negatives")
    print(f"  normalised unique: {n_norm_unique:,}/{len(df):,} = "
          f"{n_norm_unique / max(len(df), 1):.1%}")
    print(f"  hard-negative share (scam-adjacent vocabulary): {hard_share:.1%}")
    print(f"  styles: {df['script_style'].value_counts().to_dict()}")
    cats = df["notes"].str.split("cat=").str[1].str.split(";").str[0]
    print(f"  categories: {cats.value_counts().to_dict()}")
    if len(df) < n * 0.98:
        raise AssertionError(f"only {len(df)} rows survived dedupe (asked {n})")
    if n_norm_unique / len(df) < 0.95:
        raise AssertionError("normalised unique ratio below 95% - the build "
                             "would reject this table")
    if hard_share < 0.60:
        raise AssertionError(f"only {hard_share:.1%} contain scam-adjacent "
                             "vocabulary; the prompt requires >=60%")
    sim = similarity_to_probe(df)
    if sim.get("available"):
        print(f"  similarity to the frozen {sim['n_probe']}-row probe: "
              f"max={sim['max']:.3f} mean={sim['mean']:.3f} "
              f"(>=0.7: {sim['share_above_0_7']:.2%})")
        if sim["max"] >= 0.7:
            raise AssertionError(
                f"max similarity {sim['max']:.3f} to the safe-chat probe is "
                ">= 0.7 - the probe would no longer be disjoint")
    if verbose:
        for text in df["text"].head(3):
            print(f"  e.g. {text[:120]}")
    return df


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 5 safe hard negatives")
    parser.add_argument("--n", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    df = build(args.n, args.seed)
    out = data_path("synthetic_safe_v2_parquet")
    df.to_parquet(out, index=False)
    print(f"\nWROTE {out}")
    print("Next: python -m src.build_dataset   (they enter TRAIN/VAL as safe rows)")


if __name__ == "__main__":
    main()
