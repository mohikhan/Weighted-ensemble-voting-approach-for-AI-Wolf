
from __future__ import print_function, division
from extracting import *
import aiwolfpy
import aiwolfpy.contentbuilder as cb
import logging, json
from random import randint

#Saving the training data in log file named "training data.log"
myname = 'training data'.format(randint(0,99)) 

class SampleAgent(object):
    
    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        logging.basicConfig(filename=self.myname+".log",
                            level=logging.DEBUG,
                            format='')

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting):
        # New game init:
        # Store my own ID:
       
        self.myid = base_info["agentIdx"]

        self.day_no = base_info["day"]
        self.counter_negative=[0]*1 # Counter of the number of negative sentences
        self.counter_positive=[0]*1 # Counter of the number of positive sentences
        self.negative_length=[0]*1
        self.counter_negative[0] = 0
        self.counter_positive[0] = 0
        self.negative_length[0] = 0
  

        # logging.debug("# INIT: I am agent {}".format(self.myid))
        self.player_total = game_setting["playerNum"] 

        # Initialize a list with the hate score for each player
        # Also reduce own-hate score by 10k
        self.player_score = [0]*self.player_total
        self.player_score[self.myid-1] = -10000



        # My target is the agent i am targeting to predict 
        self.mytarget=1  #targeting agent 1

        self.isdead = 0 # Variable to know if my target player is dead or alive
        self.me_dead =0 # Variable to know if I am alive or not


        # the hate attribute contains the player ID that I hate the most.
        self.hate = self.player_score.index(max(self.player_score)) + 1

    # I will vote, attack and divine the most hated player so far.
    def vote(self):
        # logging.debug("# VOTE: "+str(self.hate))
        return self.hate

    def attack(self):
        # logging.debug("# ATTACK: "+str(self.hate))
        return self.hate

    def divine(self):
        # logging.debug("# DIVINE: "+str(self.hate))
        return self.hate

    def update(self, base_info, diff_data, request):
        # logging.debug("# UPDATE")
        # logging.debug("The type of update is : {}".format(request))
        
        self.day_no = base_info["day"]  # day number

    

        # Check each line in Diff Data for talks or votes
        # logging.debug(diff_data)
        for row in diff_data.itertuples():
            type = getattr(row,"type")
            text = getattr(row,"text")
            if (type == "vote"):
                voter = getattr(row,"idx")
                target = getattr(row,"agent")


                # Generating data in the log file
                
                if( self.isdead == 0 and self.me_dead == 0):
                    # If the voter is my target agent then save the data into the log file
                    if voter == self.mytarget:

                        #If the agent voted for me then save no of negative, positive sentences and 1
                        if target == self.myid:
                            logging.debug("{}  , {}  , {}  , {}  ".format(self.counter_negative[0],self.counter_positive[0],self.negative_length[0],1))
                            # logging.debug("The FINAL value of counter +ve is {}".format(self.counter_positive[0]))
                            # logging.debug("Yes my target voted for me at the end of the day!!")
              
                        #If the agent voted for me then save no of negative, positive sentences and 0
                        else :
                            
                            logging.debug("{}  , {}  , {}  , {}  ".format(self.counter_negative[0],self.counter_positive[0],self.negative_length[0],0))
                            # logging.debug("The FINAL value of counter +ve is {}".format(self.counter_positive[0]))
                            # logging.debug("No my target didnt voted for me at the end of the day!!")
              

                


                if target == self.myid:
                    
                    # logging.debug("THe Me is {} and voter is {} and the main is {}".format(self.myid,voter,self.mytarget))
                    # # logging.debug("Agent {} voted for me!".format(voter))
                    # if voter == self.mytarget:
                    #     self.votetrack=1
                    #     logging.debug("Yes he is my target {}".format(1))
                    # # else:
                    # #     logging.debug("No he is my target {}".format(0))
                    self.player_score[voter-1] += 100

                #     logging.debug("No he is my target {}".format(0))

            elif (type == "talk" and "[{:02d}]".format(self.myid) in text):
                # they are talking about me
                source = getattr(row,"agent")
                
                # logging.debug("Agent {} Talking about me: {}".format(source,text))
                if "WEREWOLF" in text or "VOTE" in text:
                    # Are you calling me a werewolf!?
                    # Are you threateningto vote me?
                    self.player_score[source-1] += 10
                else:
                    # Stop talking about me!
                    self.player_score[source-1] += 1


        # if(request == "DAILY_FINISH"):

        #     if(self.day_no!=0):
                # logging.debug("The day is finished**********************")
                # if(self.me_dead != 1 and self.isdead != 1):
                #     logging.debug("The FINAL value of counter -ve is {}".format(self.counter_negative[0]))
                #     logging.debug("The FINAL value of counter +ve is {}".format(self.counter_positive[0]))
              

        # At the beginning of the day, reduce score of dead players

        if (request == "DAILY_INITIALIZE"):

            #Initializing variables everyday
            self.counter_negative[0]=0
            self.counter_positive[0]=0
            self.negative_length[0]=0

            for i in range(self.player_total):
                if (base_info["statusMap"][str(i+1)] == "DEAD"):
                    self.player_score[i] -= 10000
                    if(base_info["statusMap"][str(1)] == "DEAD"):  #Change the value of target agent here
                        # logging.debug(" HE IS DEAD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        self.isdead = 1
                    if(base_info["statusMap"][str(self.myid)] == "DEAD"):
                        # logging.debug(" I AM DEAD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        self.me_dead = 1



        extract_update(base_info,diff_data,request,self.myid,self.counter_negative,self.counter_positive,self.negative_length,self.mytarget)
        # Print Current Hate list:
        self.hate = self.player_score.index(max(self.player_score)) + 1
        # logging.debug("Hate Score: "+", ".join(str(x) for x in self.player_score))

    # Start of the day (no return)
    def dayStart(self):
        # logging.debug("# DAYSTART")
        return None
        
    def talk(self):
        # logging.debug("# TALK")
        hatecycle = [
        "REQUEST ANY (VOTE Agent[{:02d}])",
        "ESTIMATE Agent[{:02d}] WEREWOLF",
        "VOTE Agent[{:02d}]",
        ]
        return hatecycle[randint(0,2)].format(self.hate)

    def whisper(self):
        # logging.debug("# WHISPER")
        return "ATTACK Agent[{:02d}]".format(self.hate)


    def guard(self):
        # logging.debug("# GUARD")
        return self.myid

    # Finish (no return)
    def finish(self):
        # logging.debug("# FINISH")
        # logging.debug("-------------------------------------------------GAME ENDS---------------------------------------------------------------------------------")
        return None

agent = SampleAgent(myname)

# run
if __name__ == '__main__':
    logging.debug("negative talks,positive talks,Negative length,Vote(Yes/No)") # Attributes name
    aiwolfpy.connect_parse(agent)