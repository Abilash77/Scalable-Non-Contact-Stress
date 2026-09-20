// Mental Arithmetic Task Component

import React, {Component} from 'react';

import Ticker from '../Sounds/Ticker.mp3';
import Wrong from '../Sounds/Wrong.mp3';

import MentalArithmeticAnswers from '../Layout-Components/LabStudyMentalArithmeticsAnswers';

const mentalArithmeticTasks = {
    "PR": [
        {task: "8 - 7", solution: 1},
        {task: "5 + 2", solution: 7},
        {task: "1 - 1", solution: 0},
        {task: "2 + 2 + 2", solution: 6},
        {task: "7 - 1 + 2", solution: 8},
    ],
    "LS": [
        {task: "3 - 1", solution: 2},
        {task: "4 + 3", solution: 7},
        {task: "10 - 4", solution: 6},
        {task: "3 + 3", solution: 6},
        {task: "4 + 4", solution: 8},
        {task: "5 - 1", solution: 4},
        {task: "1 + 1", solution: 2},
        {task: "3 + 2", solution: 5},
        {task: "5 + 4", solution: 9},
        {task: "6 - 1", solution: 5},
        {task: "9 - 9", solution: 0},
        {task: "0 + 1", solution: 1},
        {task: "2 + 5", solution: 7},
        {task: "8 - 4", solution: 4},
        {task: "2 + 2", solution: 4},
        {task: "5 + 3", solution: 8},
        {task: "9 - 1", solution: 8},
        {task: "7 + 2", solution: 9},
        {task: "8 - 4", solution: 4},
        {task: "6 - 3", solution: 3},
        {task: "10 - 2", solution: 8},
        {task: "1 + 5", solution: 6},
        {task: "4 + 1", solution: 5},
        {task: "3 + 4", solution: 7},
        {task: "5 - 3", solution: 2},
    ],
    "HS": [
        {task: "12 + 6 - 9 ", solution: 7},
        {task: "21 - 10 - 1", solution: 9},
        {task: "-7 + 15 - 4 + 2", solution: 6},
        {task: "17 - 18 + 6 - 1", solution: 4},
        {task: "4 - 8 - 8 + 12", solution: 0},
        {task: "19 - 9 - 4", solution: 6},
        {task: "14 - 5 - 2", solution: 7},
        {task: "19 - 9 - 9 + 6", solution: 7},
        {task: "4 + 13 - 4 - 5", solution: 8},
        {task: "16 - 8 + 7 - 12", solution: 3},
        {task: "5 - 10 + 6", solution: 1},
        {task: "3 - 8 + 10", solution: 5},
        {task: "11 + 7 - 8 - 8", solution: 2},
        {task: "5 + 3 - 10 + 7", solution: 5},
        {task: "2 - 4 - 7 + 13", solution: 4},
        {task: "5 - 6 + 3", solution: 2},
        {task: "9 + 9 - 12", solution: 5},
        {task: "5 - 4 + 12 - 13", solution: 0},
        {task: "4 + 14 - 4 - 8", solution: 6},
        {task: "9 - 7 + 8 - 9", solution: 1},
        {task: "3 + 16 - 13", solution: 6},
        {task: "15 - 8 - 7", solution: 0},
        {task: "8 + 4 - 9 + 5", solution: 8},
        {task: "14 + 1 + 2 - 8", solution: 9},
        {task: "7 + 7 - 17 + 3", solution: 0},
    ]
};

const taskContainerStyle = {
    display: "flex",
    alignItems: "center",
    flexDirection: "column",
    justifyContent: "center",
    height: "450px",
};

const scoreStyle = {
    fontSize: "20px",
};

const equationStyle = {
    fontSize: "30px",
    fontWeight: "bold"
};

const loadingBarStyle = {
    width: "500px",
    marginBottom: "20px",
    marginTop: "50px",
};

export default class MentalArithmetic extends Component {

    constructor(props) {
        super(props);
        this.state = {
            taskCount: 0,
            equation: "Prepare yourself",
            solution: "",
            score: this.props.score,
            isActive: false,
            progressValue: 200,
            progressStyle: "progress is-medium is-dark",
            modal: "modal is-active"
        };
        this.answerLogged = this.answerLogged.bind(this);
        this.initializeTask = this.initializeTask.bind(this);

        this.ticker = new Audio(Ticker);
        this.wrong = new Audio(Wrong);

        this.tasks = mentalArithmeticTasks[this.props.phaseName.slice(0,2)].slice(this.props.taskNumber, this.props.taskNumber + 5);
    }

    // Start Countdown for Task Start
    componentDidMount() {

        window.scrollTo(0, 0);

        // Start the task without the instruction if its not the practice condition
        if (this.props.phaseName.slice(0,2) !== "PR") {
            this.initializeTask()
        }

        // store the answers of the MAT task
        this.answers = [];
        this.taskNumber = this.props.taskNumber
    }

    componentWillUnmount() {
        clearTimeout(this.initiate);
        clearTimeout(this.showModal);
        clearInterval(this.progressTimer);
        clearTimeout(this.end);
        this.ticker.pause();
    }


    initializeTask() {
        this.setState({
            modal: "modal"
        });

        this.initiate = setTimeout(
            () => {
                this.setState({isActive: true});
                this.startMAT();
            },
            3000
        );
    }

    // Create Answer Options in a Circle
    createAnswerOptions(answerOpts) {
        // Angle Depending on the number of answer options
        const angle = 360 / answerOpts;
        let currAngle = 0;
        // Answer array that gets rendered
        let answers = [];
        // Creates an answer Option (now Div) and pushes it into the answer array
        for (let i = 0; i < answerOpts; i++) {
            answers.push(<MentalArithmeticAnswers angle={currAngle} key={i} num={i}
                                                  onClick={() => this.answerLogged(i)}/>);
            currAngle = currAngle + angle;
        }
        return answers
    }

    // Starts (and restarts) the Mental Arithmetic Task
    startMAT() {

        // If there are tasks remaining, update the equation, solution and taskCounter
        if (this.state.taskCount < this.tasks.length) {
            this.setState((state) => ({
                equation: this.tasks[state.taskCount]["task"],
                solution: this.tasks[state.taskCount]["solution"],
                taskCount: state.taskCount + 1
            }));
            // Set Interval to decrement the loading bar
            this.progressTimer = setInterval(
                () => this.tick(),
                10
            );
            // If there are no remaining tasks, disable pressing on answer options and end task
        } else {
            this.setState({
                equation: "Fertig",
                isActive: false
            });
            // Update the Score in the Navigator to save the Arithmetics Score and reuse it for the next trial
            this.props.updateScore(this.state.score, this.props.phaseName.slice(0,2), this.props.taskNumber + 5);
            // LabStudyEnd the task after a short timeout
            this.end = setTimeout(
                () => this.props.proceedPhase(this.props.phaseName, this.answers),
                750
            );
        }
    }

    // Decreases the loading bar value and logs wrong answer when loading is finished
    tick() {

        // Play the ticker in the high stress condition
        if (this.props.phaseName.slice(0,2) === "HS") {
            this.ticker.play();
        }

        this.setState({
            progressValue: this.state.progressValue - (100 / (5000 / 25))
        });
        if (this.state.progressValue > 100) {
            this.setState(({
                progressStyle: "progress is-medium is-dark",
            }));
        } else if (this.state.progressValue > 0) {
            this.setState(({
                progressStyle: "progress is-medium is-danger",
            }));
        } else {
            this.answerLogged(99);
        }
    }

    // Gets called when an answer is given (or no answer was given before the end of the timeout)
    answerLogged(num) {
        //Stop the audio files from playing and reset them
        this.wrong.pause();
        this.wrong.currentTime = 0;
        this.ticker.pause();
        this.ticker.currentTime = 0;
        // Only works when its active
        if (this.state.isActive) {
            // clear the started timeout
            clearInterval(this.progressTimer);
            // save answer and solution as variables
            const answer = num;
            const solution = this.state.solution;

            // play a sound if answer is wrong in the high stress condition
            if (this.props.phaseName.slice(0,2) === "HS") {
               if (answer !== solution) {
                   this.wrong.play();
               }
            }

            // console.log("Answer: " + answer, "Solution: " + solution);
            // compare the answer and the solution and change score depending on whether answer was correct or false
            this.setState((state) => ({
                score: answer === solution ? state.score + 1 : state.score - 1,
                progressValue: 200,
                progressStyle: "progress is-medium is-dark",
            }), () => {
                //Push the answer, solution and score in answers array for firebase
                this.answers.push({
                    task: this.state.equation,
                    answer: answer,
                    solution: solution,
                    score: this.state.score,
                    taskNumber: this.taskNumber,
                    time: Date.now(),
                });
                // increment the taskNumber counter
                this.taskNumber ++;
                // start new Trial
                this.startMAT();
            });
        }
    }

    // Task Instruction that gets rendered in the practice condition
    renderModal(){
        return (
            <div className={this.state.modal} style={{textAlign: "left", fontSize: "18px"}}>
                <div className="modal-background">{null}</div>
                <div className="modal-content">
                    <header className="modal-card-head">
                        <p className="modal-card-title">{null}</p>
                    </header>
                    <section className="modal-card-body">
                        <div className="content">
                            <h3>Practice task "Solving Equations"</h3>
                            <p>
                                In this task, you have to click on the correct answer to a presented equation.
                            </p>
                            <p>
                                The answer always is a number between 0 and 9. For each equation, you will see all numbers between 0 and 9 as possible solution.
                            </p>
                            <p> Above the equation, you will see your score. Clicking on the correct answer adds one point to the score and clicking on the wrong answer subtracts one point from the score.
                            </p>
                            <p>
                                For each equation, there is a time limit visualized by a loading bar.
                            </p>
                        </div>
                    </section>
                    <footer className="modal-card-foot">
                        <button className="button is-link" onClick={this.initializeTask}>Start</button>
                    </footer>
                </div>
            </div>
        )
    }

    render() {
        return (
            <div className="section">
                <div style={{height: "550px", width: "800px", border: "solid 3px black"}}>
                    <div style={taskContainerStyle}>
                        <div style={scoreStyle}>
                            <h2>Score: {this.state.score}</h2>
                        </div>
                        <div style={equationStyle}>
                            {this.state.equation}
                        </div>
                        <div>
                            {this.createAnswerOptions(10)}
                        </div>
                    </div>
                    <div style={{
                        display: "flex",
                        alignItems: "center",
                        flexDirection: "column",
                        justifyContent: "center",
                    }}>
                        <progress style={loadingBarStyle} className={this.state.progressStyle}
                                  value={this.state.progressValue} max="200">{null}</progress>
                    </div>
                </div>
                {this.props.phaseName.slice(0,2) === "PR" ? this.renderModal() : null}
            </div>
        );
    }
}