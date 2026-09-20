// Typing task component

import React, { Component } from 'react';

import KeyboardMouseTracker from "../../General-Components/KeyboardMouseTracker";
import Timer from '../Layout-Components/LabStudyTimer';

const cardStyle = {
    width: "700px",
    marginTop: "15px",
};

const typingText = {
    "PR": ["339540", "442590", "563757"],
    "HS": ["141707", "724748", "269220", "341282", "399185", "520068", "937462"],
    "LS": ["141707", "724748", "269220", "341282", "399185", "520068", "937462"]
};

export default class LabStudyPatternTyping extends Component {

    constructor(props) {
        super(props);
        this.inputRef = React.createRef();
        this.state = {
            timer: false,
            textIsCorrect: true,
            textNumber: 0,
            modal: "modal is-active"
        };

        this.renderText = this.renderText.bind(this);
        this.endTask = this.endTask.bind(this);
        this.checkIfEqual = this.checkIfEqual.bind(this);
        this.initializeTask = this.initializeTask.bind(this);
        this.checkInput = this.checkInput.bind(this);

        // Test to retype
        this.typeText = typingText[this.props.phaseName.slice(0,2)];
    }

    componentDidMount() {
        window.scrollTo(0, 0);

        // setup variable to hold store mousekeyboard data
        this.mouseKeyboardData = [];

        if (this.props.phaseName.slice(0,2) !== "PR") {
            this.initializeTask();
        }
    }

    initializeTask() {
        this.setState({
            modal: "modal",
            timer: true
        });

        this.inputRef.current.focus();
    }


    checkIfEqual (event) {
        // get the value and length of the written text in the text field
        let writtenText = event.target.value.trim();
        let textLen = writtenText.length;

        if (writtenText === this.typeText[this.state.textNumber]) {
            // if the written text is the same as the text to retype, show next text or end the task
            if (this.state.textNumber < this.typeText.length - 1) {
                this.setState({
                    textIsCorrect: true,
                    textNumber: this.state.textNumber + 1
                });
                event.target.value = "";
            } else {
                this.endTask();
            }
        } else if (this.typeText[this.state.textNumber].slice(0, textLen) !== writtenText) {
            // if the the text is the same as the sliced text to retype, show no error
            this.setState({
                textIsCorrect: false,
            })
        } else {
            // if the user makes a typing mistake, show error --> set state to incorrect
            this.setState({
                textIsCorrect: true,
            })
        }

    }

    // Only allows to type in numbers on the numpad and the backspace key
    checkInput(e) {
        // check if its a numpad key or backspace key and prevent the event if it is not
        if (!(!(e.keyCode < 48 || e.keyCode > 57) || e.keyCode === 8)) {
            e.preventDefault();
        }
    }


    renderText() {
        return(this.typeText[this.state.textNumber])
    }

    endTask(){
        clearTimeout(this.end);
        let data = {mouseKeyboardData: this.mouseKeyboardData}

        this.props.proceedPhase(this.props.phaseName, data);
    }

    onKeyboardMouseEvent(datapoint) {
        // get datapoint from tracker and fill it with page relevant info before pushing it into the data array
        const pageInfo = {
            page: this.props.phaseName,
            isCorrect: this.state.textIsCorrect,
            taskNumber: this.state.textNumber
        };

        Object.assign(datapoint, pageInfo);
        this.mouseKeyboardData.push(datapoint);
    }

    renderInstructionModal(){
        return (
            <div className={this.state.modal} style={{textAlign: "left", fontSize: "18px"}}>
                <div className="modal-background">{null}</div>
                <div className="modal-content">
                    <header className="modal-card-head">
                        <p className="modal-card-title">{null}</p>
                    </header>
                    <section className="modal-card-body">
                        <div className="content">
                            <h3>Practice task "Number Typing"</h3>
                            <p>
                                In this task, you have to correctly type in a given number.
                            </p>
                            <p>
                                You have to use the numbers above the letters on the keyboard to type in the numbers. You have to use the backspace-key to correct a mistake. Please type with your right hand only.
                            </p>
                            <p> If you make a mistake, you will receive a notification. Please correct the mistake using the backspace-key before continuing the task. As soon as you correctly typed in the number, a new number will appear.
                            </p>
                            <p>
                                The task ends after you have successfully typed in all numbers or after the countdown has expired
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
        return(
            <div>
                <KeyboardMouseTracker onEvent={this.onKeyboardMouseEvent.bind(this)}/>
                <div className="card" style={cardStyle}>
                    <header className="card-header">
                        <p className="card-header-title">
                          Remaining time:&nbsp;{!this.state.timer ? this.props.phaseName.slice(0,2) === "PR" ? "25" : "60" : <Timer time={this.props.phaseName.slice(0,2) === "PR" ? 25 : 60} end={()=> this.endTask()}/>}
                        </p>
                    </header>
                    <div className="card-content" style={{marginTop: "10px"}}>
                        <div className="content">
                            <p style={{fontSize: "26px",
                                WebkitUserSelect: "none",
                                MozUserSelect: "none",
                                msUserSelect: "none",
                                UserSelect: "none"}}>
                                {this.renderText()}
                            </p>
                        </div>
                        <div className="field">
                            <p style={{color: "hsl(348, 100%, 61%)",
                                visibility: this.state.textIsCorrect ? "hidden" : "visible"}}>
                                Please correct the mistake.
                            </p>
                            <div className="control">
                                <input style={{display: "block", margin: "auto", width: "115px", textAlign: "left"}}
                                       ref={this.inputRef}
                                       onKeyDown={this.checkInput}
                                       onKeyUp={this.checkIfEqual}
                                       className={this.state.textIsCorrect ? "input is-large is-link" : "input is-large is-danger"}/>
                            </div>
                        </div>
                    </div>
                </div>
                {this.props.phaseName.slice(0,2) === "PR" ? this.renderInstructionModal() : null}
            </div>
        );
    }

}