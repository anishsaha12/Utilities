import email
import imaplib

# class EmailReader:

#     def __init__(self, contact):                #to login contact
#         self.contact = contact
#         # Logging in to the inbox
#         self.mail = imaplib.IMAP4_SSL('imap-mail.outlook.com')
#         self.mail.login(self.contact['id'], self.contact['password'])
#         self.mail.list()
#         self.mail.select("inbox") # connect to inbox.

#         result, data = self.mail.uid('search', None, "ALL") # search and return uids instead
#         self.email_uids = data[0].split()
#         print (self.email_uids)

#     def get_decoded_email_body(self, message_body):
#         """ Decode email body.
#         Detect character set if the header is not set.
#         We try to get text/plain, but if there is not one then fallback to text/html.
#         :param message_body: Raw 7-bit message body input e.g. from imaplib. Double encoded in quoted-printable and latin-1
#         :return: Message body as unicode string
#         """

#         msg = email.message_from_string(message_body.decode('utf8'))

#         text = ""
#         if msg.is_multipart():
#             html = None
#             for part in msg.walk():

#                 # print ("%s, %s" % (part.get_content_type(), part.get_content_charset()))

#                 if part.get_content_charset() is None:
#                     # We cannot know the character set, so return decoded "something"
#                     text = part.get_payload(decode=True)
#                     continue

#                 charset = part.get_content_charset()

#                 if part.get_content_type() == 'text/plain':
#                     text = str(part.get_payload(decode=True), str(charset), "ignore").encode('utf8', 'replace')

#                 if part.get_content_type() == 'text/html':
#                     html = str(part.get_payload(decode=True), str(charset), "ignore").encode('utf8', 'replace')

#             if text is not None:
#                 return text.strip()
#             else:
#                 return html.strip()
#         else:
#             text = str(msg.get_payload(decode=True), msg.get_content_charset(), 'ignore').encode('utf8', 'replace')
#             return text.strip()

#     def get_email(self, index =-1):             #default gets latest email
#         mail = self.mail
#         email_uid = self.email_uids[index]
#         result, data = mail.uid('fetch', email_uid, '(RFC822)')
#         raw_email = data[0][1]

#         # print(raw_email.decode('utf8'))
#         email_message = email.message_from_string(raw_email.decode('utf8') )

#         email_obj = dict()
#         from_contact = dict()
#         email_from = email.utils.parseaddr(email_message['From'])   # for parsing "FirstName LastName" <email@domain.com>
#         from_contact['name']= email_from[0]
#         from_contact['id']= email_from[1]
#         from_contact['password']= '****'
#         email_obj['from'] = from_contact
#         email_obj['to'] = email_message['To'].split(',')
#         email_obj['cc'] = email_message['Cc'].split(',')
#         email_obj['date'] = email_message['Date']
#         # email_obj['content-type'] = email_message['Content-Type']
#         # email_obj['mime-version'] = email_message['MIME-Version']
#         email_obj['subject'] = email_message['Subject']
#         email_obj['body'] = self.get_decoded_email_body(raw_email)

#         return email_obj

#     def all_emails(self):
#         emails = []
#         for i in range(len(self.email_uids)):
#             emails.append(self.get_email(i))

#         return emails



# contact = { 'name':'Anish Saha',
#             'id': 'anish.saha@42hertz.com',
#             'password': '42Hertz@123'}

# my_email = EmailReader(contact)

# # with open('email_obj.txt', 'a') as the_file:
# #     the_file.write(str(my_email.all_emails()))
# print('\n\n\n',my_email.get_email(42))


def get_new_mail(last_uid = 0, host= "outlook.office365.com", port=993, login="anish.saha@42hertz.com", password="42Hertz@123"):
    print('hi')
    # connect
    mail_server = imaplib.IMAP4_SSL(host, port)

    # authenticate
    mail_server.login(login, password)

    # issue the search command of the form "SEARCH UID 42:"
    command = "UID {}:".format(last_uid)
    mail_server.select(mailbox='INBOX')
    result, data = mail_server.uid('search', None, command) # gives all uids from last_uid to latest in INBOX
    messages = data[0].split()
    print(messages)

    # yield mails
    for message_uid in messages:
        # SEARCH command always returns at least the most
        # recent message, even if it has already been synced
        if int(message_uid) > last_uid:
            result, msg = mail_server.uid('fetch', message_uid, '(RFC822)') # return a tuple with status as first and msg as second
            yield (int(message_uid), email.message_from_string(msg[0][1].decode('utf-8')))
