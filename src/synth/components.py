"""
src/synth/components.py - the Lego pieces ScamLens scam messages are built from.

Every component is a small dict made by C():

    C("english text", "roman hinglish text", "devanagari text", ["tag1", ...])
      -> {"text": {"english": ..., "roman_hinglish": ..., "devanagari": ...},
          "tactics": [...]}

Text may contain the placeholders {ENTITY} (an impersonated organisation,
filled in by the generator) and {AMOUNT} (a fake money amount).  Real URLs,
phone numbers, UPI ids and names are NEVER allowed here - only the angle
bracket placeholders <URL>, <PHONE>, <UPI> may appear inside demand texts.

A "template family" (e.g. kyc_f2) is ONE coherent way of writing a particular
scam: it has its own hooks and demands - that is what makes it a family - and
it shares the global greeting/closing pools.  Every family declares the
tactic tags it implies.  The Phase 2 splitter (src/splits2.py) holds out
whole families into test_b, so we can measure "does the model catch a scam
wording it has NEVER seen?".

HARD RULES FOR THIS FILE:
  * every tactic in src/config.py:TACTIC_TAGS has >= 4 families
  * every component carries the tactic tags it implies
  * three styles per component: english / roman_hinglish / devanagari
    (devanagari wording is kept shorter - that is allowed)

Run:  python -m src.synth.components   (prints a coverage self-check)
"""
from __future__ import annotations

from src.config import TACTIC_TAGS

# The three writing styles a component can come in.
STYLES = ("english", "roman_hinglish", "devanagari")


def C(en: str, roman: str, deva: str, tactics: list[str]) -> dict:
    """Build one component dict with the same meaning in three styles."""
    return {
        "text": {"english": en, "roman_hinglish": roman, "devanagari": deva},
        "tactics": list(tactics),
    }


# ---------------------------------------------------------------------------
# GREETINGS - shared by every family (35% of rows get no greeting at all,
# which the generator handles separately).
# ---------------------------------------------------------------------------
GREETINGS = [
    C("Dear Customer,", "Priya Grahak,", "प्रिय ग्राहक,", []),
    C("Dear Account Holder,", "Dear Account User,", "डियर अकाउंट होल्डर,", []),
    C("Hello,", "Namaste ji,", "नमस्ते जी,", []),
    C("Attention customer,", "Grahak dhyan dein,", "ध्यान दें,", []),
]

# ---------------------------------------------------------------------------
# IMPERSONATED ENTITIES - the organisation the scam pretends to be.
# A family declares which group it draws from; the generator picks one and
# substitutes it into {ENTITY}.
# ---------------------------------------------------------------------------
ENTITIES = {
    "banks": [
        "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
        "Kotak Mahindra Bank", "PNB", "IDFC First Bank", "Yes Bank",
        "Bank of Baroda", "Canara Bank",
    ],
    "wallets": [
        "Paytm", "PhonePe", "Google Pay", "BHIM", "Mobikwik",
    ],
    "telecom": [
        "Airtel", "Jio", "Vi", "BSNL",
    ],
    "courier": [
        "India Post", "DTDC", "Blue Dart", "Delhivery", "Ekart Logistics",
    ],
    "utility": [
        "the electricity board", "BSES Rajdhani", "MSEDCL", "the water board",
        "the gas agency",
    ],
    "authority": [
        "the police", "CBI", "the customs department", "TRAI",
        "the income tax department", "RBI",
    ],
    "hr": [
        "our HR team", "the recruitment team", "our HR department",
    ],
    "brand": [
        "Lucky Draw Department", "the rewards team", "Big Bazaar",
        "Flipkart", "Amazon", "the promo team",
    ],
    "broker": [
        "our trading app", "Zerodha Support", "our investment desk",
        "Groww Support",
    ],
    "loanapp": [
        "KreditKash", "MoneyTap", "the loan department", "InstaLoan",
    ],
    "none": [
        "",
    ],
}

# ---------------------------------------------------------------------------
# CLOSINGS - shared signature lines (35% of rows get no closing).
# ---------------------------------------------------------------------------
CLOSINGS = [
    C("Regards, {ENTITY} Team", "Regards, {ENTITY} Team", "धन्यवाद,", []),
    C("- {ENTITY}", "- {ENTITY}", "- {ENTITY}", []),
    C("Thank you,\\n{ENTITY}", "Dhanyawad,\\n{ENTITY}", "धन्यवाद,", []),
    C("This is an important alert from {ENTITY}.", "Yeh {ENTITY} ka zaroori alert hai.", "{ENTITY} का अलर्ट।", []),
    C("Warm regards, {ENTITY} Support", "Shubhkamnayein, {ENTITY} Support", "सादर, {ENTITY} सहायता", []),
    C("Please act at the earliest.", "Kripya jaldi karein.", "कृपया शीघ्र कार्रवाई करें।", []),
    C("For help, contact our helpline.", "Madad ke liye hamari helpline par call karein.", "सहायता के लिए हेल्पलाइन पर संपर्क करें।", []),
    C("Do not reply to this message.", "Is message ka jawab na dein.", "इस संदेश का उत्तर न दें।", []),
    C("This is an automated notification.", "Yeh ek automated soochna hai.", "यह एक स्वचालित सूचना है।", []),
    C("Keep this message for your records.", "Is message ko sambhal kar rakhein.", "इस संदेश को सुरक्षित रखें।", []),
    C("Reply STOP to unsubscribe.", "Unsubscribe karne ke liye STOP bhejein.", "अनसब्सक्राइब के लिए STOP भेजें।", []),
    C("We appreciate your quick response.", "Aapke turant jawab ke liye dhanyawad.", "आपके त्वरित उत्तर के लिए धन्यवाद।", []),
    C("Thank you for banking with us.", "Hamare saath banking ke liye dhanyawad.", "हमारे साथ जुड़े रहने के लिए धन्यवाद।", []),
    C("Sent by {ENTITY} customer care.", "{ENTITY} customer care ki taraf se.", "{ENTITY} ग्राहक सेवा की ओर से।", []),
    C("Please treat this as final.", "Ise final notice maanein.", "इसे अंतिम सूचना मानें।", []),
    C("We look forward to your response.", "Aapke jawab ka intezaar hai.", "आपके उत्तर की प्रतीक्षा है।", []),
    C("For urgent help call the number below.", "Turant madad ke liye neeche diye number par call karein.", "तत्काल सहायता के लिए नीचे दिए नंबर पर कॉल करें।", []),
    C("Issued by the {ENTITY} service desk.", "{ENTITY} service desk dwara jaari.", "{ENTITY} सेवा डेस्क द्वारा जारी।", []),
    C("Your cooperation is appreciated.", "Aapke sahyog ke liye dhanyawad.", "आपके सहयोग के लिए धन्यवाद।", []),
    C("Please do the needful.", "Kripya aavashyak karye karein.", "कृपया आवश्यक कार्य करें।", []),
]

# ---------------------------------------------------------------------------
# CONTEXTS and TAILS - neutral filler lines.
#
# WHY THEY EXIST: Phase 2.5 measured only 45% unique synthetic texts. The
# generator had ~2 hooks x ~2 demands x 4 greetings x 4 closings per family,
# so 23 rows per family repeated themselves. These pools add many harmless
# combination slots WITHOUT changing what the message asks for. A scam still
# has exactly the same hook/demand/tactics; only the surrounding wording
# varies, the way real scammers vary their templates.
# ---------------------------------------------------------------------------
CONTEXTS = [
    C("Please read this message carefully.", "Kripya is message ko dhyan se padhein.", "कृपया इस संदेश को ध्यान से पढ़ें।", []),
    C("This notice is for your registered mobile number.", "Yeh soochna aapke registered mobile number ke liye hai.", "यह सूचना आपके पंजीकृत मोबाइल नंबर के लिए है।", []),
    C("Our system generated this alert automatically.", "Hamare system ne yeh alert khud bheja hai.", "हमारे सिस्टम ने यह अलर्ट स्वयं भेजा है।", []),
    C("Kindly review the details given below.", "Neeche diye gaye details check karein.", "कृपया नीचे दिए विवरण की जाँच करें।", []),
    C("This is a time-sensitive service notice.", "Yeh ek samay-saveen service notice hai.", "यह एक समय-संवेदनशील सेवा सूचना है।", []),
    C("We are updating our records today.", "Hum aaj apne records update kar rahe hain.", "हम आज अपने रिकॉर्ड अपडेट कर रहे हैं।", []),
    C("Your recent activity needs verification.", "Aapki recent activity verify karni hai.", "आपकी हाल की गतिविधि सत्यापित करनी है।", []),
    C("Please keep your details handy.", "Apni details paas rakhein.", "कृपया अपना विवरण तैयार रखें।", []),
    C("We tried to contact you earlier.", "Humne pehle bhi aapse sampark kiya tha.", "हमने पहले भी आपसे संपर्क किया था।", []),
    C("This applies to your account immediately.", "Yeh aapke account par turant lagu hota hai.", "यह आपके खाते पर तुरंत लागू होता है।", []),
    C("A quick action from you is required.", "Aapki taraf se ek chhota action chahiye.", "आपकी ओर से एक छोटी कार्रवाई आवश्यक है।", []),
    C("Please note the reference details.", "Reference details note kar lein.", "कृपया संदर्भ विवरण नोट करें।", []),
    C("This message is important for you.", "Yeh message aapke liye zaroori hai.", "यह संदेश आपके लिए महत्वपूर्ण है।", []),
    C("Our team is reviewing your request.", "Hamari team aapki request dekh rahi hai.", "हमारी टीम आपका अनुरोध देख रही है।", []),
    C("You are receiving this as a valued customer.", "Yeh aapko ek valued customer ke roop mein mila hai.", "यह आपको एक मूल्यवान ग्राहक के रूप में मिला है।", []),
    C("Please complete the step mentioned below.", "Neeche diya gaya step poora karein.", "कृपया नीचे बताया गया चरण पूरा करें।", []),
    C("We need one confirmation from your side.", "Humein aapki taraf se ek confirmation chahiye.", "हमें आपकी ओर से एक पुष्टि चाहिए।", []),
    C("This is your final reminder for today.", "Yeh aaj ka aakhri reminder hai.", "यह आज का अंतिम अनुस्मारक है।", []),
    C("Your request is waiting for approval.", "Aapki request approval ka intezaar kar rahi hai.", "आपका अनुरोध स्वीकृति की प्रतीक्षा में है।", []),
    C("Kindly respond before the end of the day.", "Din khatam hone se pehle jawab dein.", "दिन समाप्त होने से पहले उत्तर दें।", []),
    C("We have an update regarding your case.", "Aapke case ke baare mein ek update hai.", "आपके मामले के बारे में एक अपडेट है।", []),
    C("Please check the information below.", "Neeche di gayi jaankari check karein.", "कृपया नीचे दी गई जानकारी देखें।", []),
    C("This service message needs your attention.", "Is service message par dhyan dein.", "इस सेवा संदेश पर ध्यान दें।", []),
    C("Your registered details are being checked.", "Aapki registered details check ho rahi hain.", "आपका पंजीकृत विवरण जाँचा जा रहा है।", []),
]

TAILS = [
    C("Please act now to avoid interruption.", "Rukawat se bachne ke liye abhi karein.", "रुकावट से बचने के लिए अभी कार्रवाई करें।", []),
    C("Do not share this message with anyone.", "Yeh message kisi ke saath share na karein.", "यह संदेश किसी के साथ साझा न करें।", []),
    C("This offer closes soon.", "Yeh offer jaldi band ho jayega.", "यह ऑफ़र जल्दी बंद हो जाएगा।", []),
    C("Reply YES to continue.", "Jaari rakhne ke liye YES reply karein.", "जारी रखने के लिए YES उत्तर दें।", []),
    C("Call us back for more information.", "Zyada jaankari ke liye wapas call karein.", "अधिक जानकारी के लिए वापस कॉल करें।", []),
    C("Your response is required today.", "Aapka jawab aaj hi chahiye.", "आपका उत्तर आज ही आवश्यक है।", []),
    C("Please ignore if already completed.", "Agar ho chuka hai to ignore karein.", "यदि पूरा हो चुका है तो अनदेखा करें।", []),
    C("Keep your phone nearby.", "Apna phone paas rakhein.", "अपना फ़ोन पास रखें।", []),
    C("We are available 24 hours.", "Hum 24 ghante available hain.", "हम 24 घंटे उपलब्ध हैं।", []),
    C("This is not a promotional message.", "Yeh promotional message nahi hai.", "यह प्रचार संदेश नहीं है।", []),
    C("Standard charges may apply.", "Standard charges lag sakte hain.", "मानक शुल्क लागू हो सकते हैं।", []),
    C("Please confirm at your earliest.", "Kripya jaldi confirm karein.", "कृपया शीघ्र पुष्टि करें।", []),
    C("Your account remains active for now.", "Aapka account abhi active hai.", "आपका खाता अभी सक्रिय है।", []),
    C("We regret the inconvenience.", "Asuvidha ke liye khed hai.", "असुविधा के लिए खेद है।", []),
    C("Follow the official process only.", "Sirf official process follow karein.", "केवल आधिकारिक प्रक्रिया अपनाएं।", []),
    C("Thank you for your patience.", "Aapke sabr ke liye dhanyawad.", "आपके धैर्य के लिए धन्यवाद।", []),
    C("Details will expire in 24 hours.", "Details 24 ghante mein expire hongi.", "विवरण 24 घंटे में समाप्त हो जाएंगे।", []),
    C("Please do not delay the process.", "Process mein deri na karein.", "प्रक्रिया में देरी न करें।", []),
]

# ---------------------------------------------------------------------------
# SAFE filler lines. These are used ONLY for legitimate messages, to make
# them as varied as the scams. They deliberately sound like normal service
# messages (no ask, no threat).
# ---------------------------------------------------------------------------
SAFE_CONTEXTS = [
    C("This is an automated message, please do not reply.", "Yeh automated message hai, reply na karein.", "यह स्वचालित संदेश है, उत्तर न दें।", []),
    C("For queries, call the number printed on your card.", "Sawaal ke liye card par chhapa number call karein.", "प्रश्नों के लिए कार्ड पर छपे नंबर पर कॉल करें।", []),
    C("No action is needed if this was you.", "Agar yeh aapne kiya tha to koi action ki zaroorat nahi.", "यदि यह आपने किया है तो कोई कार्रवाई आवश्यक नहीं।", []),
    C("You can also check this in the official app.", "Yeh official app mein bhi dekh sakte hain.", "आप इसे आधिकारिक ऐप में भी देख सकते हैं।", []),
    C("Please keep this message for your records.", "Kripya is message ko sambhal kar rakhein.", "कृपया इस संदेश को सुरक्षित रखें।", []),
    C("This update is sent to all registered users.", "Yeh update sabhi registered users ko bheja gaya hai.", "यह अपडेट सभी पंजीकृत उपयोगकर्ताओं को भेजा गया है।", []),
    C("Timings may vary by location.", "Samay jagah ke hisaab se badal sakta hai.", "समय स्थान के अनुसार बदल सकता है।", []),
    C("Please carry a photo ID if visiting the office.", "Office aate samay photo ID saath laayein.", "कार्यालय आते समय फोटो आईडी साथ लाएँ।", []),
    C("Thank you for choosing our service.", "Hamari service chunne ke liye dhanyawad.", "हमारी सेवा चुनने के लिए धन्यवाद।", []),
    C("Standard terms and conditions apply.", "Standard niyam aur shartein lagu hain.", "मानक नियम और शर्तें लागू हैं।", []),
    C("A copy has been sent to your email.", "Ek copy aapke email par bheji gayi hai.", "एक प्रति आपके ईमेल पर भेजी गई है।", []),
    C("No payment is required for this message.", "Is message ke liye koi payment nahi chahiye.", "इस संदेश के लिए कोई भुगतान आवश्यक नहीं।", []),
    C("Please check the official website for details.", "Details ke liye official website dekhein.", "विवरण के लिए आधिकारिक वेबसाइट देखें।", []),
    C("This is a routine service update.", "Yeh ek routine service update hai.", "यह एक नियमित सेवा अपडेट है।", []),
    C("Contact the local office if you need help.", "Madad chahiye to local office se sampark karein.", "सहायता चाहिए तो स्थानीय कार्यालय से संपर्क करें।", []),
]

SAFE_TAILS = [
    C("Have a nice day.", "Aapka din shubh ho.", "आपका दिन शुभ हो।", []),
    C("Thanks for your cooperation.", "Sahyog ke liye dhanyawad.", "सहयोग के लिए धन्यवाद।", []),
    C("Please plan accordingly.", "Kripya is hisaab se plan karein.", "कृपया तदनुसार योजना बनाएं।", []),
    C("We hope this helps.", "Aasha hai yeh madad karega.", "आशा है यह सहायक होगा।", []),
    C("Reply only if a change is needed.", "Sirf badlav ki zaroorat ho to reply karein.", "केवल बदलाव की आवश्यकता हो तो उत्तर दें।", []),
    C("Stay safe and take care.", "Surakshit rahein aur apna dhyan rakhein.", "सुरक्षित रहें और अपना ध्यान रखें।", []),
    C("See you soon.", "Jaldi milte hain.", "जल्द मिलते हैं।", []),
    C("No reply needed.", "Koi jawab zaroori nahi.", "कोई उत्तर आवश्यक नहीं।", []),
    C("Happy to help anytime.", "Kabhi bhi madad ke liye taiyar.", "कभी भी सहायता के लिए तैयार।", []),
    C("Please ignore if already done.", "Agar ho chuka hai to ignore karein.", "यदि हो चुका है तो अनदेखा करें।", []),
]

# Extra ordinary chat lines (safe, no scam vocabulary) mixed into easy safe
# families. They make casual Hinglish/English safe messages as varied as the
# scams, which removes the "style means scam" shortcut.
EASY_CHATTER = [
    C("Kal ka plan confirm hai na?", "Kal ka plan confirm hai na?", "कल का प्लान कन्फर्म है ना?", []),
    C("Main ghar pahunch gaya, tension mat lo.", "Main ghar pahunch gaya, tension mat lo.", "मैं घर पहुँच गया, चिंता मत करो।", []),
    C("Chai peene chalte hain shaam ko.", "Chai peene chalte hain shaam ko.", "शाम को चाय पीने चलते हैं।", []),
    C("Photos bhej dena jab time mile.", "Photos bhej dena jab time mile.", "समय मिले तो फोटो भेज देना।", []),
    C("Aaj office se jaldi nikal raha hoon.", "Aaj office se jaldi nikal raha hoon.", "आज ऑफिस से जल्दी निकल रहा हूँ।", []),
    C("Mummy ne poocha tum kab aa rahe ho.", "Mummy ne poocha tum kab aa rahe ho.", "मम्मी ने पूछा तुम कब आ रहे हो।", []),
    C("Match dekh rahe ho? Score kya hai?", "Match dekh rahe ho? Score kya hai?", "मैच देख रहे हो? स्कोर क्या है?", []),
    C("Bhool gaya tha, kal ka class 9 baje hai.", "Bhool gaya tha, kal ka class 9 baje hai.", "भूल गया था, कल की क्लास 9 बजे है।", []),
    C("Dinner ban gaya, aa jao.", "Dinner ban gaya, aa jao.", "डिनर बन गया, आ जाओ।", []),
    C("Tumhara phone switch off tha kya?", "Tumhara phone switch off tha kya?", "तुम्हारा फोन स्विच ऑफ था क्या?", []),
    C("Weekend pe ghar aa rahe ho?", "Weekend pe ghar aa rahe ho?", "वीकेंड पर घर आ रहे हो?", []),
    C("Doctor ne kya bola, sab theek hai?", "Doctor ne kya bola, sab theek hai?", "डॉक्टर ने क्या कहा, सब ठीक है?", []),
    C("New phone aa gaya, number same hai.", "New phone aa gaya, number same hai.", "नया फोन आ गया, नंबर वही है।", []),
    C("Grocery list bana li? Main le aata hoon.", "Grocery list bana li? Main le aata hoon.", "किराने की लिस्ट बना ली? मैं ले आता हूँ।", []),
    C("Bus late hai, 20 minute lagenge.", "Bus late hai, 20 minute lagenge.", "बस लेट है, 20 मिनट लगेंगे।", []),
    C("Meeting ke notes bhej do please.", "Meeting ke notes bhej do please.", "मीटिंग के नोट्स भेज दो प्लीज़।", []),
    C("Landlord ne rent reminder bheja hai.", "Landlord ne rent reminder bheja hai.", "मकान मालिक ने किराया रिमाइंडर भेजा है।", []),
    C("Kal subah walk pe chalte hain.", "Kal subah walk pe chalte hain.", "कल सुबह वॉक पर चलते हैं।", []),
    C("Tickets book ho gaye, screenshot bhej raha hoon.", "Tickets book ho gaye, screenshot bhej raha hoon.", "टिकट बुक हो गए, स्क्रीनशॉट भेज रहा हूँ।", []),
    C("Aunty ka tabiyat ab better hai.", "Aunty ka tabiyat ab better hai.", "आंटी की तबीयत अब बेहतर है।", []),
    C("Please pick up some milk on the way home.", "Please pick up some milk on the way home.", "घर आते समय दूध ले आना।", []),
    C("Parcel guard ke paas rakh diya hai.", "Parcel guard ke paas rakh diya hai.", "पार्सल गार्ड के पास रख दिया है।", []),
    C("Kal ki meeting reschedule ho gayi hai.", "Kal ki meeting reschedule ho gayi hai.", "कल की मीटिंग रीशेड्यूल हो गई है।", []),
    C("Lunch kya bana hai? Bhookh lagi hai.", "Lunch kya bana hai? Bhookh lagi hai.", "लंच क्या बना है? भूख लगी है।", []),
]


# ---------------------------------------------------------------------------
# TEMPLATE FAMILIES - part 1 of 2.
# One entry per family: name, tactic tags, entity group, hooks, demands.
# tactics in the family list = the baseline tags; hooks/demands add more
# (e.g. a demand containing <URL> also implies suspicious_link).
# ---------------------------------------------------------------------------
FAMILIES: list[dict] = [
    # -----------------------------------------------------------------------
    # otp_or_pin_request (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "otp_f1",
        "tactics": ["otp_or_pin_request"],
        "entity_group": "banks",
        "hooks": [
            C("A login was detected on your {ENTITY} account from a new device.",
              "Aapke {ENTITY} account par naye device se login hua hai.",
              "{ENTITY} अकाउंट में नए डिवाइस से लॉगिन हुआ।", ["urgency_threat"]),
            C("Unusual sign-in attempt on your {ENTITY} account.",
              "Aapke {ENTITY} account par ajeeb sign-in aaya hai.",
              "{ENTITY} अकाउंट पर अजीब साइन-इन।", ["urgency_threat"]),
        ],
        "demands": [
            C("Share the OTP you just received so we can block the login.",
              "Jo OTP aaya hai wo share karein taaki hum login block kar dein.",
              "आपको OTP साझा करें।", ["otp_or_pin_request"]),
            C("Confirm your OTP to keep your account safe.",
              "Account safe rakhne ke liye OTP confirm karein.",
              "अकाउंट सुरक्षित रखने के लिए OTP पुष्टि करें।", ["otp_or_pin_request"]),
        ],
    },
    {
        "name": "otp_f2",
        "tactics": ["otp_or_pin_request"],
        "entity_group": "wallets",
        "hooks": [
            C("Your {ENTITY} KYC is pending and will stop working today.",
              "Aapka {ENTITY} KYC pending hai aur aaj band ho jayega.",
              "{ENTITY} KYC पेंडिंग है।", ["bank_kyc_impersonation", "urgency_threat"]),
            C("We could not verify your {ENTITY} wallet.",
              "Hum aapka {ENTITY} wallet verify nahi kar paye.",
              "{ENTITY} वॉलेट सत्यापित नहीं हुआ।", []),
        ],
        "demands": [
            C("Enter the OTP sent to your mobile to re-verify.",
              "Dobara verify karne ke liye mobile par aaya OTP daalein.",
              "OTP दर्ज करें।", ["otp_or_pin_request"]),
            C("Verify with OTP now or your wallet will be frozen.",
              "Abhi OTP se verify karein warna wallet freeze ho jayega.",
              "अभी OTP सत्यापित करें।", ["otp_or_pin_request", "urgency_threat"]),
        ],
    },
    {
        "name": "otp_f3",
        "tactics": ["otp_or_pin_request"],
        "entity_group": "telecom",
        "hooks": [
            C("Your SIM will be deactivated in 2 hours as KYC is incomplete.",
              "Aapki SIM 2 ghante mein band ho jayegi, KYC adhoori hai.",
              "आपकी SIM बंद हो जाएगी।", ["utility_disconnection_scam", "urgency_threat"]),
            C("Final notice: your {ENTITY} number is unverified.",
              "Aakhri notice: aapka {ENTITY} number unverified hai.",
              "आखिरी नोटिस।", ["urgency_threat"]),
        ],
        "demands": [
            C("Reply with the OTP to reactivate your SIM immediately.",
              "SIM turant chalu karne ke liye OTP reply karein.",
              "SIM सक्रिय करने के लिए OTP भेजें।", ["otp_or_pin_request"]),
            C("Share OTP to complete SIM re-verification.",
              "SIM re-verify karne ke liye OTP share karein.",
              "OTP साझा करें।", ["otp_or_pin_request"]),
        ],
    },
    {
        "name": "otp_f4",
        "tactics": ["otp_or_pin_request"],
        "entity_group": "banks",
        "hooks": [
            C("A refund of Rs {AMOUNT} is pending in your {ENTITY} account.",
              "Aapke {ENTITY} account mein Rs {AMOUNT} ka refund pending hai.",
              "रिफंड पेंडिंग है।", ["payment_or_upi_request"]),
            C("Your card refund of Rs {AMOUNT} is ready to credit.",
              "Aapke card ka Rs {AMOUNT} refund credit hone ko taiyar hai.",
              "कार्ड रिफंड तैयार है।", []),
        ],
        "demands": [
            C("To release the refund, first confirm the OTP on your number.",
              "Refund ke liye number par OTP confirm karein.",
              "रिफंड के लिए OTP पुष्टि करें।", ["otp_or_pin_request"]),
            C("Validate the transaction with the OTP we sent you.",
              "Bheje gaye OTP se transaction validate karein.",
              "OTP से लेनदेन सत्यापित करें।", ["otp_or_pin_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # urgency_threat (4 families) - fear and a clock, no real ask besides speed
    # -----------------------------------------------------------------------
    {
        "name": "urg_f1",
        "tactics": ["urgency_threat"],
        "entity_group": "banks",
        "hooks": [
            C("Suspicious transaction of Rs {AMOUNT} on your {ENTITY} card.",
              "Aapke {ENTITY} card par Rs {AMOUNT} ka suspicious transaction hua.",
              "संदिग्ध लेनदेन हुआ।", []),
            C("Your account will be blocked today for non-compliance.",
              "Aaj aapka account block ho jayega.",
              "आज अकाउंट ब्लॉक होगा।", []),
        ],
        "demands": [
            C("Call the number below within 30 minutes to stop the block.",
              "Block rokne ke liye 30 minute mein neeche diya number call karein.",
              "कॉल करें।", ["authority_impersonation"]),
            C("Do not ignore this, act now or account will be blocked.",
              "Ise ignore mat karein, abhi warna account block.",
              "अभी कार्य करें।", []),
        ],
    },
    {
        "name": "urg_f2",
        "tactics": ["urgency_threat"],
        "entity_group": "authority",
        "hooks": [
            C("Your name appears in a cyber crime complaint filed today.",
              "Aapka naam aaj cyber crime complaint mein hai.",
              "आपका नाम शिकायत में है।", ["authority_impersonation"]),
            C("An arrest warrant has been issued in your name.",
              "Aapke naam par giraftari warrant jaari hui hai.",
              "गिरफ्तारी वारंट जारी हुई है।", ["authority_impersonation"]),
        ],
        "demands": [
            C("Contact us immediately or legal action starts today.",
              "Turant sampark karein warna aaj kanooni karyawahi shuru.",
              "तुरंत संपर्क करें।", ["authority_impersonation"]),
            C("Cooperate now to avoid being declared a fugitive.",
              "Abhi sahyog karein warna bhagghar ghoshit.",
              "अभी सहयोग करें।", ["authority_impersonation"]),
        ],
    },
    {
        "name": "urg_f3",
        "tactics": ["urgency_threat"],
        "entity_group": "courier",
        "hooks": [
            C("Your parcel is held by customs and will be returned today.",
              "Aapka parcel customs mein atka hai aur aaj wapas jayega.",
              "पार्सल अटका है।", ["parcel_courier_scam"]),
            C("Parcel delivery failed, package marked return to sender.",
              "Parcel deliver nahi hua, wapas bhej diya jayega.",
              "पार्सल वापस जाएगा।", ["parcel_courier_scam"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} clearance charges in the next 2 hours.",
              "Agle 2 ghante mein Rs {AMOUNT} clearance charges bhejein.",
              "चार्ज भेजें।", ["payment_or_upi_request"]),
            C("Clear the pending charges now to receive your parcel.",
              "Parcel paane ke liye pending charges abhi clear karein.",
              "बकाया चार्ज अदा करें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "urg_f4",
        "tactics": ["urgency_threat"],
        "entity_group": "utility",
        "hooks": [
            C("Your electricity connection will be disconnected TONIGHT.",
              "Aaj raat aapki bijli connection kat jayegi.",
              "आज रात बिजली कट जाएगी।", ["utility_disconnection_scam"]),
            C("Disconnection notice: unpaid bill for last 2 months.",
              "Katne ka notice: pichle 2 mahine ka bill unpaid hai.",
              "भुगतान नहीं हुआ।", ["utility_disconnection_scam"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} today to avoid disconnection.",
              "Aaj Rs {AMOUNT} bhejein warna connection kat jayega.",
              "आज भुगतान करें।", ["payment_or_upi_request"]),
            C("Recharge now with Rs {AMOUNT} or supply stops at 8 pm.",
              "Abhi Rs {AMOUNT} recharge karein warna 8 baje bijli band.",
              "अभी रिचार्ज करें।", ["payment_or_upi_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # bank_kyc_impersonation (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "kyc_f1",
        "tactics": ["bank_kyc_impersonation"],
        "entity_group": "banks",
        "hooks": [
            C("Your {ENTITY} KYC has expired as per RBI guidelines.",
              "RBI guidelines ke mutabik aapka {ENTITY} KYC expire ho gaya hai.",
              "आपका KYC समाप्त हो गया है।", ["authority_impersonation"]),
            C("Update your KYC else your account will be frozen.",
              "KYC update karein warna account freeze ho jayega.",
              "KYC अपडेट करें।", ["urgency_threat"]),
        ],
        "demands": [
            C("Click <URL> and update your KYC immediately.",
              "Turant <URL> par click karke KYC update karein.",
              "<URL> पर क्लिक करें।", ["suspicious_link"]),
            C("Update KYC here: <URL>",
              "Yahan KYC update karein: <URL>",
              "यहाँ KYC अपडेट करें: <URL>", ["suspicious_link"]),
        ],
    },
    {
        "name": "kyc_f2",
        "tactics": ["bank_kyc_impersonation"],
        "entity_group": "banks",
        "hooks": [
            C("Your {ENTITY} account is marked incomplete KYC.",
              "Aapka {ENTITY} account incomplete KYC ka marked hai.",
              "अकाउंट अधूरा KYC है।", []),
            C("Re-KYC due for your account, take action today.",
              "Aapke account ka Re-KYC due hai, aaj karein.",
              "Re-KYC आज करें।", ["urgency_threat"]),
        ],
        "demands": [
            C("Fill your details at <URL> before midnight.",
              "Raat 12 se pehle <URL> par details bharein.",
              "मध्यरात्रि से पहले भरें।", ["suspicious_link"]),
            C("Submit Aadhaar and PAN online at <URL>",
              "Aadhaar aur PAN online <URL> par bhejein.",
              "आधार और पैन भेजें।", ["suspicious_link"]),
        ],
    },
    {
        "name": "kyc_f3",
        "tactics": ["bank_kyc_impersonation"],
        "entity_group": "wallets",
        "hooks": [
            C("{ENTITY} account will stop working without full KYC.",
              "Poori KYC ke bina {ENTITY} band ho jayega.",
              "पूरा KYC नहीं तो वॉलेट बंद।", ["urgency_threat"]),
            C("Your {ENTITY} mini-KYC verification failed.",
              "Aapka {ENTITY} mini-KYC verify nahi hua.",
              "मिनी-KYC सत्यापित नहीं हुआ।", []),
        ],
        "demands": [
            C("Complete KYC by sending Rs {AMOUNT} to our verify UPI <UPI>.",
              "KYC complete karne ke liye Rs {AMOUNT} <UPI> par bhejein.",
              "<UPI> पर भेजें।", ["payment_or_upi_request"]),
            C("Send a small amount to <UPI> to finish verification.",
              "Verification ke liye <UPI> par thoda amount bhejein.",
              "<UPI> पर थोड़ा पैसा भेजें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "kyc_f4",
        "tactics": ["bank_kyc_impersonation"],
        "entity_group": "banks",
        "hooks": [
            C("Documents pending for your {ENTITY} account since long.",
              "Aapke {ENTITY} account ke documents lambe samay se pending hain.",
              "दस्तावेज़ पेंडिंग हैं।", []),
            C("Your branch has flagged missing KYC papers.",
              "Aapki branch ne KYC papers missing flag kiye hain.",
              "ब्रांच ने फ्लैग किया।", []),
        ],
        "demands": [
            C("Verify your documents on this call, dial <PHONE> now.",
              "Abhi <PHONE> par call karke documents verify karein.",
              "अभी <PHONE> पर कॉल करें।", ["authority_impersonation"]),
            C("Our agent will call you from <PHONE> for KYC.",
              "Hamara agent <PHONE> se KYC ke liye call karegaa.",
              "एजेंट <PHONE> से कॉल करेगा।", []),
        ],
    },

    # -----------------------------------------------------------------------
    # authority_impersonation (4 families) - police/customs/tax/TRAI
    # -----------------------------------------------------------------------
    {
        "name": "auth_f1",
        "tactics": ["authority_impersonation"],
        "entity_group": "authority",
        "hooks": [
            C("This is {ENTITY}. Your Aadhaar is linked to a drug case.",
              "Hum {ENTITY} hain. Aapka Aadhaar ek drug case se juda hai.",
              "आपका आधार केस से जुड़ा है।", ["urgency_threat"]),
            C("CBI notice: your documents were found at a crime scene.",
              "CBI notice: aapke documents crime scene se mile hain.",
              "दस्तावेज़ बरामद हुए।", ["urgency_threat"]),
        ],
        "demands": [
            C("Join this video call for digital arrest proceedings.",
              "Digital arrest ke liye video call join karein.",
              "वीडियो कॉल जॉइन करें।", ["urgency_threat"]),
            C("Do not tell anyone, appear online or a warrant is issued.",
              "Kisi ko mat bataiye, online pesh ho ya warrant aayega.",
              "किसी को न बताएं।", ["urgency_threat"]),
        ],
    },
    {
        "name": "auth_f2",
        "tactics": ["authority_impersonation"],
        "entity_group": "authority",
        "hooks": [
            C("Customs: a parcel in your name contains illegal cash.",
              "Customs: aapke naam par parcel mein illegal cash hai.",
              "पार्सल में अवैध नकद है।", ["parcel_courier_scam", "urgency_threat"]),
            C("Your passport application is under investigation.",
              "Aapke passport application ki jaanch chal rahi hai.",
              "पासपोर्ट आवेदन की जांच चल रही है।", []),
        ],
        "demands": [
            C("Transfer your bank balance to the safe account given here.",
              "Diye gaye safe account mein balance transfer karein.",
              "सेफ अकाउंट में ट्रांसफर करें।", ["payment_or_upi_request"]),
            C("Shift your money to the government holding account now.",
              "Abhi apna paisa sarkari holding account mein daalein.",
              "पैसा सरकारी खाते में डालें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "auth_f3",
        "tactics": ["authority_impersonation"],
        "entity_group": "telecom",
        "hooks": [
            C("TRAI alert: your number will be blocked for spam messages.",
              "TRAI alert: spam ke liye aapka number block hoga.",
              "आपका नंबर ब्लॉक होगा।", ["utility_disconnection_scam", "urgency_threat"]),
            C("Your mobile numbers are being disconnected for misuse.",
              "Galat istemal ke liye aapke numbers band ho rahe hain.",
              "नंबर बंद हो रहे हैं।", ["utility_disconnection_scam"]),
        ],
        "demands": [
            C("Press 1 on the call to stay connected.",
              "Juday rehne ke liye call mein 1 dabayein.",
              "कॉल में 1 दबाएं।", []),
            C("Confirm your number details on the call from <PHONE>.",
              "<PHONE> se call par number details confirm karein.",
              "<PHONE> से कॉल पर पुष्टि करें।", []),
        ],
    },
    {
        "name": "auth_f4",
        "tactics": ["authority_impersonation"],
        "entity_group": "authority",
        "hooks": [
            C("Income tax dept: your bank accounts are under scrutiny.",
              "Income tax dept: aapke bank accounts ki scrutiny chal rahi hai.",
              "खातों की जांच चल रही है।", ["urgency_threat"]),
            C("You have unfiled taxes; a case is being prepared.",
              "Aapke unfiled taxes hain, case ban raha hai.",
              "केस बन रहा है।", ["urgency_threat"]),
        ],
        "demands": [
            C("Pay the settlement of Rs {AMOUNT} to close the case.",
              "Case band karne ke liye Rs {AMOUNT} settlement bhejein.",
              "सेटलमेंट भेजें।", ["payment_or_upi_request"]),
            C("Deposit Rs {AMOUNT} in the refund account to clear dues.",
              "Dues clear karne ke liye Rs {AMOUNT} refund account mein jama karein.",
              "जमा करें।", ["payment_or_upi_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # payment_or_upi_request (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "upi_f1",
        "tactics": ["payment_or_upi_request"],
        "entity_group": "wallets",
        "hooks": [
            C("You have a pending UPI collect request of Rs {AMOUNT}.",
              "Aapka Rs {AMOUNT} ka UPI collect request pending hai.",
              "UPI कलेक्ट अनुरोध पेंडिंग है।", []),
            C("A friend is requesting Rs {AMOUNT} on {ENTITY}.",
              "Ek dost {ENTITY} par Rs {AMOUNT} maang raha hai.",
              "एक दोस्त पैसे मांग रहा है।", []),
        ],
        "demands": [
            C("Approve the request and enter your UPI PIN to receive money.",
              "Request approve karein aur paisa paane ke liye UPI PIN daalein.",
              "UPI पिन दर्ज करें।", ["otp_or_pin_request"]),
            C("Enter your UPI PIN to accept the collect request.",
              "Collect request accept karne ke liye UPI PIN daalein.",
              "UPI पिन दर्ज करें।", ["otp_or_pin_request"]),
        ],
    },
    {
        "name": "upi_f2",
        "tactics": ["payment_or_upi_request"],
        "entity_group": "none",
        "hooks": [
            C("Brother sent you Rs {AMOUNT} by mistake, please return it.",
              "Bhai ne galti se Rs {AMOUNT} bheja hai, wapas kar do.",
              "गलती से पैसे भेजे हैं।", ["relative_in_trouble"]),
            C("I sent money to your number by error, need it back.",
              "Maine galti se aapke number par paisa bheja tha.",
              "गलती से पैसे भेजे।", []),
        ],
        "demands": [
            C("Send Rs {AMOUNT} back on the UPI link <UPI>.",
              "Rs {AMOUNT} <UPI> par wapas bhej do.",
              "<UPI> पर वापस भेजें।", ["suspicious_link"]),
            C("Refund it to the same number from which it came.",
              "Jis number se aaya hai usi number par wapas bhejo.",
              "उसी नंबर पर वापस भेजें।", []),
        ],
    },
    {
        "name": "upi_f3",
        "tactics": ["payment_or_upi_request"],
        "entity_group": "none",
        "hooks": [
            C("To verify your UPI, send Rs 1 to our merchant ID.",
              "UPI verify karne ke liye Rs 1 hamare merchant ID par bhejein.",
              "UPI सत्यापित करने के लिए ₹1 भेजें।", ["bank_kyc_impersonation"]),
            C("Small verification amount needed to activate your account.",
              "Account activate karne ke liye thoda verification amount chahiye.",
              "सत्यापन राशि चाहिए।", []),
        ],
        "demands": [
            C("Pay Rs 1 now and your account will be activated instantly.",
              "Abhi Rs 1 bhejein aur account turant activate ho jayega.",
              "अभी ₹1 भेजें।", []),
            C("Send the token amount to <UPI> to continue.",
              "Jaari rakhne ke liye <UPI> par token amount bhejein.",
              "<UPI> पर राशि भेजें।", ["suspicious_link"]),
        ],
    },
    {
        "name": "upi_f4",
        "tactics": ["payment_or_upi_request"],
        "entity_group": "hr",
        "hooks": [
            C("Salary release pending, complete your payment profile first.",
              "Salary pending hai, pehle payment profile complete karein.",
              "वेतन लंबित है।", ["job_or_task_scam"]),
            C("Refund for your order is stuck, pay a small fee to release it.",
              "Aapke order ka refund atka hai, release karne ke liye fee bhejein.",
              "रिफंड अटका है।", []),
        ],
        "demands": [
            C("Transfer Rs {AMOUNT} as processing fee to <UPI>.",
              "Processing fee ke roop mein Rs {AMOUNT} <UPI> par bhejein.",
              "प्रोसेसिंग शुल्क भेजें।", ["suspicious_link"]),
            C("Pay the refundable deposit of Rs {AMOUNT} to get your money.",
              "Paisa paane ke liye refundable deposit Rs {AMOUNT} bhejein.",
              "जमा राशि भेजें।", []),
        ],
    },

    # -----------------------------------------------------------------------
    # suspicious_link (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "link_f1",
        "tactics": ["suspicious_link"],
        "entity_group": "courier",
        "hooks": [
            C("Your courier parcel could not be delivered.",
              "Aapka courier parcel deliver nahi ho paya.",
              "पार्सल डिलीवर नहीं हुआ।", ["parcel_courier_scam"]),
            C("Shipment on hold: address incomplete.",
              "Shipment atka hua hai: address adhoora hai.",
              "पता अधूरा है।", []),
        ],
        "demands": [
            C("Update your address at <URL> to receive the parcel.",
              "Parcel paane ke liye <URL> par address update karein.",
              "<URL> पर पता अपडेट करें।", []),
            C("Reschedule delivery from <URL>",
              "<URL> se delivery reschedule karein.",
              "<URL> से डिलीवरी बदलें।", []),
        ],
    },
    {
        "name": "link_f2",
        "tactics": ["suspicious_link"],
        "entity_group": "brand",
        "hooks": [
            C("Congratulations! You won Rs {AMOUNT} in our lucky draw.",
              "Badhai! Aapne humare lucky draw mein Rs {AMOUNT} jeeta hai.",
              "बधाई! आप लकी ड्रॉ जीत गए।", ["prize_or_lottery"]),
            C("You are selected for a special gift from {ENTITY}.",
              "Aap {ENTITY} ka special gift jeete hain.",
              "आपको विशेष उपहार मिला है।", ["prize_or_lottery"]),
        ],
        "demands": [
            C("Claim your prize here before it expires: <URL>",
              "Prize claim karein: <URL>",
              "पुरस्कार प्राप्त करें: <URL>", ["urgency_threat"]),
            C("Tap <URL> to see your reward.",
              "Apna dekhne ke liye <URL> par tap karein.",
              "अपना इनाम देखें: <URL>", []),
        ],
    },
    {
        "name": "link_f3",
        "tactics": ["suspicious_link"],
        "entity_group": "banks",
        "hooks": [
            C("Statement of your {ENTITY} account is attached.",
              "Aapke {ENTITY} account ka statement attached hai.",
              "खाते का विवरण संलग्न है।", []),
            C("Your updated passbook PDF is ready to view.",
              "Aapka updated passbook PDF dekhne ke liye taiyar hai.",
              "पासबुक PDF तैयार है।", []),
        ],
        "demands": [
            C("Download your statement from <URL> using your mobile number.",
              "Mobile number se <URL> se statement download karein.",
              "<URL> से डाउनलोड करें।", ["otp_or_pin_request"]),
            C("View statement: <URL>",
              "Statement dekhein: <URL>",
              "विवरण देखें: <URL>", []),
        ],
    },
    {
        "name": "link_f4",
        "tactics": ["suspicious_link"],
        "entity_group": "loanapp",
        "hooks": [
            C("Pre-approved loan of Rs {AMOUNT} is waiting for you.",
              "Rs {AMOUNT} ka pre-approved loan aapka wait kar raha hai.",
              "पहले से स्वीकृत ऋण तैयार है।", []),
            C("Instant loan offer, no documents needed.",
              "Instant loan offer, koi document nahi chahiye.",
              "तुरंत ऋण, कोई दस्तावेज़ नहीं।", []),
        ],
        "demands": [
            C("Apply from <URL> and get money in minutes.",
              "<URL> se apply karein aur minute mein paisa paayein.",
              "<URL> से आवेदन करें।", []),
            C("Check your eligibility at <URL>",
              "Apni eligibility <URL> par check karein.",
              "<URL> पर जांचें।", []),
        ],
    },
]

# ---------------------------------------------------------------------------
# TEMPLATE FAMILIES - part 2 of 2 (remaining tactic groups).
# ---------------------------------------------------------------------------
FAMILIES += [
    # -----------------------------------------------------------------------
    # prize_or_lottery (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "prize_f1",
        "tactics": ["prize_or_lottery"],
        "entity_group": "brand",
        "hooks": [
            C("Your mobile number won Rs {AMOUNT} in today's lucky draw.",
              "Aapka mobile number aaj ke lucky draw mein jeeta hai.",
              "आपका नंबर लकी ड्रॉ में जीता।", []),
            C("{ENTITY} bumper prize notification for your number.",
              "{ENTITY} ka bumper prize aapke number ke liye.",
              "आपके नंबर के लिए बंपर इनाम।", []),
        ],
        "demands": [
            C("Share your bank details to receive the prize money.",
              "Prize money paane ke liye bank details share karein.",
              "इनाम पाने के लिए बैंक विवरण भेजें।", ["payment_or_upi_request"]),
            C("Confirm your account number where the prize should be sent.",
              "Confirm karein kis account mein prize bheja jaye.",
              "खाता संख्या पुष्टि करें।", []),
        ],
    },
    {
        "name": "prize_f2",
        "tactics": ["prize_or_lottery"],
        "entity_group": "brand",
        "hooks": [
            C("You have been shortlisted for a festival cash reward.",
              "Aap festival cash reward ke liye shortlist hue hain.",
              "आप चुने गए हैं।", []),
            C("Final reminder: your gift voucher is unclaimed.",
              "Aakhri reminder: aapka gift voucher claim nahi hua.",
              "गिफ्ट वाउचर बाकी है।", ["urgency_threat"]),
        ],
        "demands": [
            C("Claim before 6 pm at <URL>",
              "Sham 6 baje se pehle <URL> par claim karein.",
              "शाम 6 से पहले क्लेम करें।", ["suspicious_link"]),
            C("Click <URL> and register your prize.",
              "<URL> par click karke prize register karein.",
              "<URL> पर क्लिक करें।", ["suspicious_link"]),
        ],
    },
    {
        "name": "prize_f3",
        "tactics": ["prize_or_lottery"],
        "entity_group": "none",
        "hooks": [
            C("Dear winner, your lottery ticket number is selected.",
              "Dear winner, aapka lottery ticket number select hua hai.",
              "आपका लॉटरी टिकट चुना गया है।", []),
            C("International lottery: you are the lucky winner today.",
              "International lottery: aaj aap lucky winner hain.",
              "अंतर्राष्ट्रीय लॉटरी: आप विजेता हैं।", []),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} as registration fee to claim your prize.",
              "Prize claim karne ke liye Rs {AMOUNT} registration fee bhejein.",
              "पंजीकरण शुल्क भेजें।", ["payment_or_upi_request"]),
            C("Transfer the tax amount of Rs {AMOUNT} to release your winnings.",
              "Jeete hue paiso ke liye Rs {AMOUNT} tax bhejein.",
              "कर राशि भेजें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "prize_f4",
        "tactics": ["prize_or_lottery"],
        "entity_group": "brand",
        "hooks": [
            C("{ENTITY} anniversary offer: you won a brand new phone!",
              "{ENTITY} anniversary offer: aapne naya phone jeeta!",
              "आपने नया फ़ोन जीता!", []),
            C("Spin the wheel and claim your guaranteed reward.",
              "Wheel ghumao aur guaranteed reward lo.",
              "घुमाएं और इनाम लें।", []),
        ],
        "demands": [
            C("Verify your delivery address at <URL>",
              "Delivery address <URL> par verify karein.",
              "पता <URL> पर सत्यापित करें।", ["suspicious_link"]),
            C("Pay Rs {AMOUNT} shipping charge so we can courier the prize.",
              "Prize courier karne ke liye Rs {AMOUNT} shipping bhejein.",
              "शिपिंग शुल्क भेजें।", ["payment_or_upi_request", "parcel_courier_scam"]),
        ],
    },

    # -----------------------------------------------------------------------
    # job_or_task_scam (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "job_f1",
        "tactics": ["job_or_task_scam"],
        "entity_group": "hr",
        "hooks": [
            C("You are selected for the data entry role you applied for.",
              "Aap data entry role ke liye select hue hain jo aapne apply kiya tha.",
              "आप चुने गए हैं।", []),
            C("{ENTITY} has an opening matching your profile.",
              "{ENTITY} mein aapke profile ke liye vacancy hai.",
              "आपके प्रोफ़ाइल के लिए वैकेंसी है।", []),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} registration fee to start work from home.",
              "Ghar se kaam shuru karne ke liye Rs {AMOUNT} registration fee bhejein.",
              "पंजीकरण शुल्क भेजें।", ["payment_or_upi_request"]),
            C("Share your Aadhaar and PAN to complete joining formalities.",
              "Joining ke liye Aadhaar aur PAN share karein.",
              "आधार और पैन भेजें।", ["authority_impersonation"]),
        ],
    },
    {
        "name": "job_f2",
        "tactics": ["job_or_task_scam"],
        "entity_group": "none",
        "hooks": [
            C("Part-time task work: earn Rs {AMOUNT} daily doing simple likes.",
              "Part-time task: roz simple likes karke Rs {AMOUNT} kamaayein.",
              "पार्ट-टास्क काम, रोज़ कमाएं।", ["investment_or_trading_scam"]),
            C("Earn money daily by rating apps. No investment needed today.",
              "Apps rating karke roz kamaayein.",
              "ऐप रेटिंग से कमाएं।", []),
        ],
        "demands": [
            C("Join our Telegram group at <URL> to receive tasks.",
              "Tasks paane ke liye Telegram group <URL> par join karein.",
              "<URL> पर जॉइन करें।", ["suspicious_link"]),
            C("Message us on WhatsApp at <PHONE> to start today's task.",
              "Aaj ka task shuru karne ke liye <PHONE> par WhatsApp karein.",
              "<PHONE> पर व्हाट्सएप करें।", []),
        ],
    },
    {
        "name": "job_f3",
        "tactics": ["job_or_task_scam"],
        "entity_group": "hr",
        "hooks": [
            C("Interview scheduled: HR will call you shortly.",
              "Interview scheduled: HR aapko jaldi call karegi.",
              "इंटरव्यू तय है।", []),
            C("Your profile is shortlisted for an MNC position.",
              "Aapka profile MNC post ke liye shortlisted hai.",
              "प्रोफ़ाइल शॉर्टलिस्ट हुई है।", []),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} for the offer letter processing.",
              "Offer letter processing ke liye Rs {AMOUNT} bhejein.",
              "प्रोसेसिंग शुल्क भेजें।", ["payment_or_upi_request"]),
            C("Buy the training kit worth Rs {AMOUNT} to confirm your seat.",
              "Seat confirm karne ke liye Rs {AMOUNT} ka training kit lein.",
              "ट्रेनिंग किट लें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "job_f4",
        "tactics": ["job_or_task_scam"],
        "entity_group": "none",
        "hooks": [
            C("Work from home vacancy: Rs {AMOUNT} per month, 2 hours daily.",
              "Ghar baithe kaam: Rs {AMOUNT} mahina, roz 2 ghante.",
              "घर बैठे काम, हर महीने।", []),
            C("Hiring immediately, no interview, guaranteed joining.",
              "Turant bharti, no interview, guaranteed joining.",
              "तुरंत भर्ती।", []),
        ],
        "demands": [
            C("Deposit Rs {AMOUNT} as security amount to get your ID card.",
              "ID card paane ke liye Rs {AMOUNT} security jama karein.",
              "सुरक्षा राशि जमा करें।", ["payment_or_upi_request"]),
            C("Complete first task at <URL> and earn today.",
              "<URL> par pehla task complete karein aur aaj kamaayein.",
              "<URL> पर पहला टास्क करें।", ["suspicious_link"]),
        ],
    },

    # -----------------------------------------------------------------------
    # investment_or_trading_scam (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "invest_f1",
        "tactics": ["investment_or_trading_scam"],
        "entity_group": "broker",
        "hooks": [
            C("Your trading account shows a profit of Rs {AMOUNT} today.",
              "Aaj aapke trading account mein Rs {AMOUNT} profit hai.",
              "ट्रेडिंग खाते में मुनाफ़ा है।", []),
            C("Your investment plan has matured with returns.",
              "Aapka investment plan returns ke saath mature hua hai.",
              "निवेश योजना परिपक्व हुई।", []),
        ],
        "demands": [
            C("Withdraw your profit by paying the Rs {AMOUNT} withdrawal charge.",
              "Profit nikalne ke liye Rs {AMOUNT} withdrawal charge bhejein.",
              "निकासी शुल्क भेजें।", ["payment_or_upi_request"]),
            C("Add more funds to <UPI> to unlock your returns.",
              "Returns unlock karne ke liye <UPI> par aur paisa daalein.",
              "<UPI> पर और पैसा जमा करें।", ["payment_or_upi_request", "suspicious_link"]),
        ],
    },
    {
        "name": "invest_f2",
        "tactics": ["investment_or_trading_scam"],
        "entity_group": "none",
        "hooks": [
            C("Double your money in 7 days with our expert tips.",
              "Hamare expert tips se 7 din mein paisa double karein.",
              "7 दिनों में पैसा दोगुना।", []),
            C("Daily income plan: Rs {AMOUNT} guaranteed every day.",
              "Daily income plan: roz Rs {AMOUNT} guaranteed.",
              "रोज़ गारंटीड आय।", []),
        ],
        "demands": [
            C("Start with Rs {AMOUNT} on <URL> and watch it grow.",
              "Rs {AMOUNT} se <URL> par shuru karein.",
              "<URL> पर शुरू करें।", ["suspicious_link"]),
            C("Register at <URL> before the offer closes.",
              "Offer band hone se pehle <URL> par register karein.",
              "<URL> पर रजिस्टर करें।", ["suspicious_link", "urgency_threat"]),
        ],
    },
    {
        "name": "invest_f3",
        "tactics": ["investment_or_trading_scam"],
        "entity_group": "broker",
        "hooks": [
            C("Bitcoin price is rising, invest now and earn big.",
              "Bitcoin ka price badh raha hai, abhi invest karein.",
              "बिटकॉइन की कीमत बढ़ रही है।", []),
            C("Our analysts predict 300% return this month.",
              "Hamare analysts is mahine bada return dete hain.",
              "इस महीने बड़े रिटर्न का अनुमान।", []),
        ],
        "demands": [
            C("Send the investment amount to <UPI> to open your demat account.",
              "Demat account kholne ke liye <UPI> par paisa bhejein.",
              "डीमैट खाते के लिए <UPI> पर भेजें।", ["payment_or_upi_request"]),
            C("Open your demat account here: <URL>",
              "Demat account yahan kholen: <URL>",
              "डीमैट खाता यहाँ खोलें: <URL>", ["suspicious_link"]),
        ],
    },
    {
        "name": "invest_f4",
        "tactics": ["investment_or_trading_scam"],
        "entity_group": "none",
        "hooks": [
            C("Congratulations on your first trading profit!",
              "Aapke pehle trading profit ki badhai!",
              "पहले ट्रेडिंग मुनाफ़े की बधाई!", []),
            C("Your withdrawal is on hold pending verification.",
              "Aapka withdrawal verification ke liye atka hai.",
              "निकासी रुकी है।", ["urgency_threat"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} tax before we can release your withdrawal.",
              "Withdrawal ke liye pehle Rs {AMOUNT} tax bhejein.",
              "पहले कर भेजें।", ["payment_or_upi_request"]),
            C("Recharge Rs {AMOUNT} to activate withdrawal channel.",
              "Withdrawal channel activate karne ke liye Rs {AMOUNT} recharge karein.",
              "निकासी चैनल सक्रिय करने के लिए रिचार्ज करें।", ["payment_or_upi_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # loan_app_harassment (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "loan_f1",
        "tactics": ["loan_app_harassment"],
        "entity_group": "loanapp",
        "hooks": [
            C("Your loan of Rs {AMOUNT} is overdue by 12 days.",
              "Aapka Rs {AMOUNT} ka loan 12 din se overdue hai.",
              "आपका ऋण 12 दिन से बकाया है।", ["urgency_threat"]),
            C("Legal notice: unpaid loan amount in your name.",
              "Kanooni notice: aapke naam par unpaid loan hai.",
              "कानूनी नोटिस: बकाया ऋण।", ["authority_impersonation", "urgency_threat"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} today or your family will be contacted.",
              "Aaj Rs {AMOUNT} bhejein warna aapke family se baat ki jayegi.",
              "आज भुगतान करें।", ["relative_in_trouble"]),
            C("Clear dues now or your photo will be shared publicly.",
              "Dues clear karein warna aapki photo sabko bheji jayegi.",
              "बकाया साफ़ करें।", ["urgency_threat"]),
        ],
    },
    {
        "name": "loan_f2",
        "tactics": ["loan_app_harassment"],
        "entity_group": "loanapp",
        "hooks": [
            C("EMI of Rs {AMOUNT} bounced from your account.",
              "Aapke account se Rs {AMOUNT} ki EMI bounce ho gayi.",
              "ईएमआई बाउंस हो गई।", []),
            C("Your loan account is flagged as defaulter.",
              "Aapka loan account defaulter flag hua hai.",
              "ऋण खाता डिफॉल्टर है।", []),
        ],
        "demands": [
            C("Pay the Rs {AMOUNT} penalty immediately on <UPI>.",
              "Turant <UPI> par Rs {AMOUNT} penalty bhejein.",
              "अभी <UPI> पर पेनल्टी भेजें।", ["payment_or_upi_request"]),
            C("Call <PHONE> to settle your loan today.",
              "Aaj apna loan settle karne ke liye <PHONE> par call karein.",
              "आज <PHONE> पर कॉल करें।", []),
        ],
    },
    {
        "name": "loan_f3",
        "tactics": ["loan_app_harassment"],
        "entity_group": "loanapp",
        "hooks": [
            C("Final recovery call before we file a police complaint.",
              "Police complaint se pehli aakhri recovery call.",
              "पुलिस शिकायत से पहले आखिरी कॉल।", ["authority_impersonation", "urgency_threat"]),
            C("Your CIBIL score will drop if payment is not made.",
              "Payment na karne par aapka CIBIL score girega.",
              "आपका CIBIL स्कोर गिरेगा।", ["urgency_threat"]),
        ],
        "demands": [
            C("Transfer Rs {AMOUNT} to avoid a court case.",
              "Court case se bachne ke liye Rs {AMOUNT} bhejein.",
              "मुकदमे से बचने के लिए भेजें।", ["payment_or_upi_request"]),
            C("Pay partial Rs {AMOUNT} now to stop recovery visits.",
              "Recovery visit rokne ke liye abhi Rs {AMOUNT} bhejein.",
              "रिकवरी विज़िट रोकने के लिए भेजें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "loan_f4",
        "tactics": ["loan_app_harassment"],
        "entity_group": "loanapp",
        "hooks": [
            C("Warning: recovery agent assigned to your case today.",
              "Chetawani: aapke case ke liye recovery agent assign hua.",
              "चेतावनी: रिकवरी एजेंट नियुक्त।", ["urgency_threat"]),
            C("Your loan closure amount has increased.",
              "Aapka loan closure amount badh gaya hai.",
              "ऋण समापन राशि बढ़ गई है।", []),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} closure fee to close the loan file.",
              "Loan file band karne ke liye Rs {AMOUNT} closure fee bhejein.",
              "समापन शुल्क भेजें।", ["payment_or_upi_request"]),
            C("Send the overdue amount to <UPI> within today.",
              "Aaj ke andar overdue amount <UPI> par bhejein.",
              "आज ही <UPI> पर भेजें।", ["payment_or_upi_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # parcel_courier_scam (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "parcel_f1",
        "tactics": ["parcel_courier_scam"],
        "entity_group": "courier",
        "hooks": [
            C("Your parcel with an illegal item is stuck at the office.",
              "Aapka parcel illegal cheez ke saath office mein atka hai.",
              "पार्सल अटक गया है।", ["authority_impersonation", "urgency_threat"]),
            C("{ENTITY} parcel needs your ID proof to clear customs.",
              "{ENTITY} parcel clear karne ke liye ID proof chahiye.",
              "पार्सल साफ़ करने के लिए आईडी चाहिए।", ["authority_impersonation"]),
        ],
        "demands": [
            C("Send Rs {AMOUNT} customs charge to release the parcel.",
              "Parcel chhutwane ke liye Rs {AMOUNT} customs charge bhejein.",
              "कस्टम शुल्क भेजें।", ["payment_or_upi_request"]),
            C("Pay the small charge of Rs {AMOUNT} on <UPI> now.",
              "Abhi <UPI> par Rs {AMOUNT} charge bhejein.",
              "अभी <UPI> पर शुल्क भेजें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "parcel_f2",
        "tactics": ["parcel_courier_scam"],
        "entity_group": "courier",
        "hooks": [
            C("Delivery attempt failed: wrong address on the package.",
              "Delivery fail: package par galat address hai.",
              "डिलीवरी असफल: गलत पता।", []),
            C("Your parcel is waiting at the depot for re-delivery.",
              "Aapka parcel depot par re-delivery ka wait kar raha hai.",
              "पार्सल डिपो में है।", []),
        ],
        "demands": [
            C("Confirm your address by calling <PHONE> immediately.",
              "Turant <PHONE> par call karke address confirm karein.",
              "अभी <PHONE> पर कॉल करें।", []),
            C("Reschedule from <URL> or the parcel returns to sender.",
              "<URL> se reschedule karein warna parcel wapas jayega.",
              "<URL> से फिर से शेड्यूल करें।", ["suspicious_link", "urgency_threat"]),
        ],
    },
    {
        "name": "parcel_f3",
        "tactics": ["parcel_courier_scam"],
        "entity_group": "courier",
        "hooks": [
            C("A parcel in your name contains foreign currency.",
              "Aapke naam par parcel mein videshi currency hai.",
              "पार्सल में विदेशी मुद्रा है।", ["authority_impersonation"]),
            C("Postal department: your package failed verification.",
              "Postal department: aapka package verify nahi hua.",
              "पैकेज सत्यापित नहीं हुआ।", ["authority_impersonation"]),
        ],
        "demands": [
            C("Reply to this message to avoid return of the parcel.",
              "Parcel wapas na bhejne ke liye is message ka reply dein.",
              "जवाब दें।", []),
            C("Send your Aadhaar details to clear the package.",
              "Package clear karne ke liye Aadhaar details bhejein.",
              "पैकेज साफ़ करने के लिए आधार भेजें।", ["authority_impersonation"]),
        ],
    },
    {
        "name": "parcel_f4",
        "tactics": ["parcel_courier_scam"],
        "entity_group": "courier",
        "hooks": [
            C("COD parcel worth Rs {AMOUNT} out for delivery today.",
              "Rs {AMOUNT} ka COD parcel aaj deliver ho raha hai.",
              "आज सीओडी पार्सल डिलीवर होगा।", []),
            C("New shipment registered to your number.",
              "Aapke number par naya shipment register hua hai.",
              "नया शिपमेंट दर्ज हुआ।", []),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} advance to confirm the COD delivery.",
              "COD confirm karne ke liye Rs {AMOUNT} advance bhejein.",
              "सीओडी के लिए एडवांस भेजें।", ["payment_or_upi_request"]),
            C("Verify by paying Rs 1 at <UPI>",
              "Verify karne ke liye <UPI> par Rs 1 bhejein.",
              "सत्यापन के लिए <UPI> पर ₹1 भेजें।", ["payment_or_upi_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # utility_disconnection_scam (4 families)
    # -----------------------------------------------------------------------
    {
        "name": "util_f1",
        "tactics": ["utility_disconnection_scam"],
        "entity_group": "utility",
        "hooks": [
            C("Electricity bill unpaid for 3 months, disconnect order issued.",
              "3 mahine ka bijli bill unpaid hai, katne ka order jaari.",
              "बिजली बिल बकाया है।", ["urgency_threat"]),
            C("Last reminder before power disconnection tonight.",
              "Aaj raat bijli katne se pehli aakhri reminder.",
              "आज रात बिजली कटने से पहले आखिरी रिमाइंडर।", ["urgency_threat"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} on <URL> to keep your power on.",
              "Bijli rakhne ke liye <URL> par Rs {AMOUNT} bhejein.",
              "<URL> पर भुगतान करें।", ["suspicious_link", "payment_or_upi_request"]),
            C("Recharge Rs {AMOUNT} before 7 pm today.",
              "Aaj shaam 7 se pehle Rs {AMOUNT} recharge karein.",
              "आज शाम 7 से पहले रिचार्ज करें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "util_f2",
        "tactics": ["utility_disconnection_scam"],
        "entity_group": "telecom",
        "hooks": [
            C("Your {ENTITY} number will be expired tomorrow.",
              "Aapka {ENTITY} number kal expire ho jayega.",
              "आपका नंबर कल समाप्त होगा।", ["urgency_threat"]),
            C("Outgoing services stopped: recharge required.",
              "Outgoing band: recharge zaroori hai.",
              "आउटगोइंग बंद: रिचार्ज ज़रूरी।", []),
        ],
        "demands": [
            C("Do Rs {AMOUNT} recharge immediately to continue services.",
              "Services jaari rakhne ke liye turant Rs {AMOUNT} recharge karein.",
              "तुरंत रिचार्ज करें।", ["payment_or_upi_request"]),
            C("Call <PHONE> to reactivate your number for free.",
              "Number free mein chalu karne ke liye <PHONE> par call karein.",
              "नंबर सक्रिय करने के लिए <PHONE> पर कॉल करें।", []),
        ],
    },
    {
        "name": "util_f3",
        "tactics": ["utility_disconnection_scam"],
        "entity_group": "utility",
        "hooks": [
            C("Gas cylinder delivery failed, payment pending.",
              "Gas cylinder delivery fail, payment pending hai.",
              "गैस सिलेंडर डिलीवरी असफल।", []),
            C("Water supply will be cut for non-payment.",
              "Pani ki supply payment na karne par kat jayegi.",
              "पानी की आपूर्ति कटेगी।", ["urgency_threat"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} on <UPI> to restore the connection.",
              "Connection wapas paane ke liye <UPI> par Rs {AMOUNT} bhejein.",
              "<UPI> पर भुगतान करें।", ["suspicious_link", "payment_or_upi_request"]),
            C("Send the bill amount to <UPI> and share the screenshot.",
              "Bill amount <UPI> par bhejein aur screenshot share karein.",
              "बिल राशि भेजें और स्क्रीनशॉट साझा करें।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "util_f4",
        "tactics": ["utility_disconnection_scam"],
        "entity_group": "telecom",
        "hooks": [
            C("KYC pending: your SIM services will stop tonight.",
              "KYC pending: aaj raat SIM band ho jayegi.",
              "KYC पेंडिंग: आज रात SIM बंद।", ["bank_kyc_impersonation", "urgency_threat"]),
            C("Aadhaar linking incomplete for your mobile number.",
              "Aapke mobile number ka Aadhaar linking adhoora hai.",
              "आधार लिंकिंग अधूरी है।", ["bank_kyc_impersonation"]),
        ],
        "demands": [
            C("Re-link now at <URL> or services stop at midnight.",
              "<URL> par abhi re-link karein warna raat 12 baje band.",
              "अभी <URL> पर लिंक करें।", ["suspicious_link"]),
            C("Share the OTP to finish Aadhaar linking.",
              "Aadhaar linking complete karne ke liye OTP share karein.",
              "आधार लिंकिंग के लिए OTP साझा करें।", ["otp_or_pin_request"]),
        ],
    },

    # -----------------------------------------------------------------------
    # relative_in_trouble (4 families) - fake son/daughter/brother messages
    # -----------------------------------------------------------------------
    {
        "name": "rel_f1",
        "tactics": ["relative_in_trouble"],
        "entity_group": "none",
        "hooks": [
            C("Hi, this is my new number. Saved it?",
              "Hi, yeh mera naya number hai. Save kar liya?",
              "हाय, यह मेरा नया नंबर है।", []),
            C("Mom, I dropped my phone, using friend's number today.",
              "Mummy, mera phone toot gaya, aaj dost ke number se likh rahi hoon.",
              "माँ, फ़ोन टूट गया।", []),
        ],
        "demands": [
            C("I need Rs {AMOUNT} urgently, please send on <UPI>.",
              "Mujhe Rs {AMOUNT} turant chahiye, <UPI> par bhej do.",
              "मुझे तुरंत चाहिए।", ["payment_or_upi_request", "urgency_threat"]),
            C("Send money now, I will return it tomorrow.",
              "Abhi paisa bhej do, kal wapas kar dungi.",
              "अभी पैसे भेजो।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "rel_f2",
        "tactics": ["relative_in_trouble"],
        "entity_group": "none",
        "hooks": [
            C("Papa, I met with an accident, I am in the hospital.",
              "Papa, main accident mein gayi hoon, abhi hospital mein hoon.",
              "पापा, मैं एक्सीडेंट में घायल हूँ।", ["urgency_threat"]),
            C("Sir, your son has been brought to the police station.",
              "Sir, aapka beta police station laya gaya hai.",
              "सर, आपका बेटा पुलिस स्टेशन में है।", ["authority_impersonation", "urgency_threat"]),
        ],
        "demands": [
            C("Pay Rs {AMOUNT} hospital fee so they treat me now.",
              "Abhi ilaj ke liye Rs {AMOUNT} hospital fee bhejo.",
              "अस्पताल शुल्क भेजो।", ["payment_or_upi_request"]),
            C("The officer is asking for Rs {AMOUNT}, please send quickly.",
              "Officer Rs {AMOUNT} maang rahe hain, jaldi bhejo.",
              "जल्दी भेजो।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "rel_f3",
        "tactics": ["relative_in_trouble"],
        "entity_group": "none",
        "hooks": [
            C("Bhai, I am stuck, phone battery is almost over.",
              "Bhai, main atka hoon, phone battery khatam ho rahi hai.",
              "भाई, मैं फँस गया हूँ।", ["urgency_threat"]),
            C("This is your brother, I changed my WhatsApp number.",
              "Main tera bhai hoon, mera WhatsApp number badal gaya hai.",
              "मैं तेरा भाई हूँ।", []),
        ],
        "demands": [
            C("Send Rs {AMOUNT} to the number I will share shortly.",
              "Rs {AMOUNT} us number par bhejo jo main abhi bhejunga.",
              "उस नंबर पर भेजो।", ["payment_or_upi_request"]),
            C("Pay the Rs {AMOUNT} fee to get me out of here.",
              "Yahan se nikalne ke liye Rs {AMOUNT} fee bhar do.",
              "फ़ीस भर दो।", ["payment_or_upi_request"]),
        ],
    },
    {
        "name": "rel_f4",
        "tactics": ["relative_in_trouble"],
        "entity_group": "none",
        "hooks": [
            C("Dear, I am messaging from the embassy, your brother is with us.",
              "Main embassy se message kar rahi hoon, aapka bhai hamare paas hai.",
              "मैं दूतावास से लिख रही हूँ।", ["authority_impersonation", "urgency_threat"]),
            C("Mom, my exam fee is due tomorrow, please help.",
              "Mummy, meri exam fee kal last hai, madad karo.",
              "माँ, मेरी एग्ज़ाम फ़ीस कल लास्ट है।", []),
        ],
        "demands": [
            C("Deposit Rs {AMOUNT} at the account number I share next.",
              "Main jo account number bhej rahi hoon usme Rs {AMOUNT} jama karo.",
              "बताए खाते में जमा करो।", ["payment_or_upi_request"]),
            C("Send Rs {AMOUNT} on <UPI> for the release paperwork.",
              "Release papers ke liye <UPI> par Rs {AMOUNT} bhejo.",
              "<UPI> पर भेजो।", ["payment_or_upi_request", "suspicious_link"]),
        ],
    },
]


# ---------------------------------------------------------------------------
# SHORT SCAM ONE-LINERS (added after the shortcut audit flagged msg_length:
# composed scams averaged ~120 chars while safe messages averaged ~58, so a
# model could "detect scams" by LENGTH alone. Real scam SMS are often very
# short, so we add terse one-liner families - one per tactic - and the
# generator mixes them in with the long composed scams.)
# Each entry is its OWN template family (own name) so the splitter can hold
# some of them out. Templates keep only <URL>/<PHONE> placeholders.
# ---------------------------------------------------------------------------
SCAM_SHORT_FAMILIES: list[dict] = [
    {"name": "otp_s1", "tactics": ["otp_or_pin_request"],
     "templates": [
         C("OTP update pending. Verify now: <URL>",
           "OTP update pending hai. Verify karein: <URL>",
           "OTP अपडेट पेंडिंग। सत्यापित करें: <URL>", ["urgency_threat"]),
         C("Your OTP is required to keep a/c active. Share it.",
           "A/c active rakhne ke liye OTP chahiye. Share karein.",
           "खाता सक्रिय रखने के लिए OTP भेजें।", []),
     ]},
    {"name": "kyc_s1", "tactics": ["bank_kyc_impersonation"],
     "templates": [
         C("KYC expired. Update: <URL>",
           "KYC expire ho gaya. Update: <URL>",
           "KYC समाप्त। अपडेट: <URL>", ["urgency_threat"]),
         C("Final notice: re-KYC today or a/c blocked.",
           "Aakhri notice: aaj re-KYC karein warna a/c block.",
           "अंतिम सूचना: आज re-KYC करें।", ["urgency_threat"]),
     ]},
    {"name": "urg_s1", "tactics": ["urgency_threat"],
     "templates": [
         C("Action needed within 2 hours or account suspended.",
           "2 ghante mein action lein warna account suspended.",
           "2 घंटे में कार्रवाई करें।", []),
         C("Last warning. Respond now.",
           "Aakhri warning. Abhi reply karein.",
           "अंतिम चेतावनी। अभी जवाब दें।", []),
     ]},
    {"name": "auth_s1", "tactics": ["authority_impersonation"],
     "templates": [
         C("Cyber cell: your no. under watch. Call <PHONE> now.",
           "Cyber cell: aapka no. watch par hai. Abhi call <PHONE>",
           "साइबर सेल: आपका नंबर निगरानी में।", ["urgency_threat"]),
         C("Customs: parcel held. Pay Rs 500 to release: <URL>",
           "Customs: parcel ruka hai. Rs 500 bhejein: <URL>",
           "कस्टम्स: पार्सल रुका। Rs 500 भेजें।", ["parcel_courier_scam", "payment_or_upi_request"]),
     ]},
    {"name": "upi_s1", "tactics": ["payment_or_upi_request"],
     "templates": [
         C("Rs 1 UPI collect request sent. Accept to verify account.",
           "Rs 1 ka UPI collect bheja hai. Account verify karne par accept karein.",
           "Rs 1 UPI कलेक्ट भेजा गया।", ["otp_or_pin_request"]),
         C("Send Rs 1 to activate your cashback.",
           "Cashback activate karne ke liye Rs 1 bhejein.",
           "कैशबैक सक्रिय करने के लिए Rs 1 भेजें।", ["prize_or_lottery"]),
     ]},
    {"name": "link_s1", "tactics": ["suspicious_link"],
     "templates": [
         C("Your file is pending. Download: <URL>",
           "Aapka file pending hai. Download: <URL>",
           "आपका फ़ाइल पेंडिंग है। डाउनलोड: <URL>", []),
         C("New document shared with you: <URL>",
           "Aapke saath naya document share hua: <URL>",
           "नया दस्तावेज़ साझा किया गया: <URL>", []),
     ]},
    {"name": "prize_s1", "tactics": ["prize_or_lottery"],
     "templates": [
         C("You won Rs 25 lakh! Claim: <URL>",
           "Aapne Rs 25 lakh jeeta! Claim: <URL>",
           "आपने Rs 25 lakh जीते! क्लेम: <URL>", ["urgency_threat"]),
         C("Lucky draw result: your number selected. Collect prize <URL>",
           "Lucky draw: aapka number chuna gaya. Prize <URL> par lein",
           "लकी ड्रॉ: आपका नंबर चुना गया।", []),
     ]},
    {"name": "job_s1", "tactics": ["job_or_task_scam"],
     "templates": [
         C("Part-time job, Rs 2,000 daily, no experience. WhatsApp <PHONE>",
           "Part-time kaam, Rs 2,000 roz. WhatsApp <PHONE> par",
           "पार्ट-टाइम काम, Rs 2,000 रोज़।", ["payment_or_upi_request"]),
         C("Task do, paisa kamao. Join now: <URL>",
           "Task karo, paisa kamao. Abhi join: <URL>",
           "टास्क करो, पैसा कमाओ।", []),
     ]},
    {"name": "inv_s1", "tactics": ["investment_or_trading_scam"],
     "templates": [
         C("Double your money in 7 days. Minimum Rs 500. <URL>",
           "7 din mein paisa double. Minimum Rs 500. <URL>",
           "7 दिनों में पैसा दोगुना।", ["payment_or_upi_request"]),
         C("Sure shot trading tip, daily profit. Join <URL>",
           "Pakka trading tip, roz profit. Join <URL> par",
           "पक्का ट्रेडिंग टिप, रोज़ प्रॉफिट।", []),
     ]},
    {"name": "loan_s1", "tactics": ["loan_app_harassment"],
     "templates": [
         C("Loan approved! Pay Rs 500 processing fee to disburse.",
           "Loan approved! Disburse ke liye Rs 500 processing fee bhejein.",
           "लोन स्वीकृत! Rs 500 प्रोसेसिंग फ़ीस भेजें।", ["payment_or_upi_request"]),
         C("Overdue instalment. Pay today or action follows.",
           "Kist overdue hai. Aaj pay karein warna action.",
           "किस्त ओवरड्यू है। आज भुगतान करें।", ["urgency_threat"]),
     ]},
    {"name": "parcel_s1", "tactics": ["parcel_courier_scam"],
     "templates": [
         C("Parcel stuck at customs. Pay Rs 45: <URL>",
           "Parcel customs mein atka hai. Rs 45 bhejein: <URL>",
           "पार्सल कस्टम्स में फँसा। Rs 45 भेजें।", ["payment_or_upi_request"]),
         C("Undelivered parcel. Rebook now: <URL>",
           "Parcel deliver nahi hua. Abhi rebook: <URL>",
           "पार्सल नहीं पहुँचा। अभी रीबुक: <URL>", ["urgency_threat"]),
     ]},
    {"name": "util_s1", "tactics": ["utility_disconnection_scam"],
     "templates": [
         C("Electricity disconnected today. Pay: <URL>",
           "Bijli aaj kat jayegi. Payment: <URL>",
           "बिजली आज कट जाएगी। भुगतान: <URL>", ["urgency_threat"]),
         C("SIM blocked. Recharge now to restore: <PHONE>",
           "SIM block ho gaya. Wapas karne ke liye recharge: <PHONE>",
           "SIM बंद हो गया। रीचार्ज करें।", ["urgency_threat"]),
     ]},
    {"name": "rel_s1", "tactics": ["relative_in_trouble"],
     "templates": [
         C("Papa hai naye number se. Paise bhejo: <UPI>",
           "Papa hoon naye number se. Paise bhejo: <UPI>",
           "पापा हूँ नए नंबर से। पैसे भेजें।", ["payment_or_upi_request"]),
         C("Beta, urgent paise bhej do, baad mein bataunga.",
           "Beta, turant paise bhej do. Baad mein batata hoon.",
           "बेटा, तुरंत पैसे भेज दो।", ["urgency_threat"]),
     ]},
]


# ---------------------------------------------------------------------------
# SMALL API used by src/synth/generate.py and the self-check below.
# ---------------------------------------------------------------------------
def all_families() -> list[dict]:
    """Every template family, in file order."""
    return list(FAMILIES)


def short_scam_families() -> list[dict]:
    """The terse one-liner scam families (length-leak fix, see above)."""
    return list(SCAM_SHORT_FAMILIES)


def families_for(tactic: str) -> list[dict]:
    """Families whose BASE tactic list contains `tactic` (direct tag)."""
    return [f for f in FAMILIES if tactic in f["tactics"]]


def all_tactics_of(family: dict) -> set[str]:
    """Every tactic a family can ever imply (family + its hooks/demands)."""
    tags = set(family["tactics"])
    for comp in family["hooks"] + family["demands"]:
        tags.update(comp["tactics"])
    return tags


def coverage_gaps() -> list[str]:
    """Tactic tags with fewer than 4 families (HARD RULE check)."""
    gaps = []
    for tag in TACTIC_TAGS:
        # Require the tag in the family's BASE list so the generator can
        # deliberately build that tactic (component tags may add more).
        if len(families_for(tag)) < 4:
            gaps.append(tag)
    return gaps


if __name__ == "__main__":
    # Tiny self-check: python -m src.synth.components
    n_comp = sum(len(f["hooks"]) + len(f["demands"]) for f in FAMILIES)
    print(f"components.py: {len(FAMILIES)} families, {n_comp} family components")
    gaps = coverage_gaps()
    if gaps:
        print("COVERAGE GAP - tactics with <4 families:", gaps)
        raise SystemExit(1)
    print("OK: every tactic in the taxonomy has >= 4 template families.")
    for tag in TACTIC_TAGS:
        names = [f["name"] for f in families_for(tag)]
        print(f"  {tag:32s} -> {', '.join(names)}")

