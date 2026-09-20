// Follow-Box Task component

import React, { Component } from 'react';

import Timer from '../Layout-Components/LabStudyTimer';
import '../Style-Sheets/Lab-MovingBox.css';

import KeyboardMouseTracker from "../../General-Components/KeyboardMouseTracker";


const taskContainerStyle = {
    height: "630px",
    width: "1100px",
    border: "solid 3px black",
    marginTop: "10px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    // backgroundColor: "#4169E1"
};


let mousePosition = {
    x: 0,
    y: 0
};

let boxPosition = {
  left: null,
  right: null,
  top: null,
  bottom: null
};

export default class LabStudyFollowBoxTask extends Component {

    constructor(props) {
        super(props);
        this.boxRef = React.createRef();
        this.state = {
            onObject: false,
            started: false,
            modal: "modal is-active"
        };

        this.startTask = this.startTask.bind(this);
        this.endTask = this.endTask.bind(this);
        this.initializeTask = this.initializeTask.bind(this);
        this.mouseInBox = this.mouseInBox.bind(this);
        this.getBoxPosition = this.getBoxPosition.bind(this);
    }

    componentDidMount() {

        window.scrollTo(0, 0);

        // setup variable to store tracker data
        this.mouseKeyboardData = [];

        if (this.props.phaseName.slice(0,2) !== "PR") {
            this.initializeTask();
        }
    }


    initializeTask() {
        this.setState({
            modal: "modal",
        });

        let boxPos = this.boxRef.current.getBoundingClientRect();
        boxPosition.left = boxPos.left;
        boxPosition.right = boxPos.right;
        boxPosition.top = boxPos.top;
        boxPosition.bottom = boxPos.bottom;

    }

    startTask(){
        this.setState({
            started: true,
            onObject: true
        });

        this.getBoxPosition();
    }

    // get the current position of the target box and check if the mouse cursor is inside it
    getBoxPosition() {
        if (this.boxRef.current) {
            let boxPos = this.boxRef.current.getBoundingClientRect();
            boxPosition.left = boxPos.left;
            boxPosition.right = boxPos.right;
            boxPosition.top = boxPos.top;
            boxPosition.bottom = boxPos.bottom;
            this.mouseInBox(mousePosition.x, mousePosition.y, boxPosition.left, boxPosition.right, boxPosition.top, boxPosition.bottom);
            requestAnimationFrame(this.getBoxPosition);

        }
    }

    // check if the mouse cursor is inside the target box
    mouseInBox(mouseX, mouseY, boxLeft, boxRight, boxTop, boxBottom) {

        if (mouseX >= boxLeft && mouseX <= boxRight
        && mouseY >= boxTop && mouseY <= boxBottom) {

            if (!this.state.started) {
                this.startTask();
            } else {
                this.setState({
                    onObject: true
                })
            }

        } else {
            this.setState({
                onObject: false
            })
        }

    }

    endTask(){

        let data = {mouseKeyboardData: this.mouseKeyboardData}

        this.props.proceedPhase(this.props.phaseName, data);
    }


    onKeyboardMouseEvent(datapoint) {

        // if its a mousemove parameter, update the stored mouseX and mouseY coordinates and check if the mouse cursor is
        // inside the target box
        if (datapoint.eventType === "MousePositionChanged") {
            mousePosition.x = datapoint.x;
            mousePosition.y = datapoint.y;
            this.mouseInBox(mousePosition.x, mousePosition.y, boxPosition.left, boxPosition.right, boxPosition.top, boxPosition.bottom);
        }

        // save mouse data and add page relevant info before pushing it into the data array
        // information about whether the task has started and if the mouse cursor is inside the box
        const pageInfo = {
            page: this.props.phaseName,
            taskStarted: this.state.started,
            inBox: this.state.onObject
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
                            <h3>Practice task "Follow-the-box"</h3>
                            <p>
                                In this task, you have to follow a moving box with your mouse cursor.
                            </p>
                            <p>
                                Try to keep your mouse cursor inside the moving box. As long as your mouse cursor is inside the box, the background color of the box is blue. As long as your mouse cursor is outside the box, the background color of the box is grey.
                            </p>
                            <p>
                                The task starts as soon as you move the mouse cursor inside the box. The task will end after the countdown has expired.
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
                <p style={{
                    WebkitUserSelect: "none",
                    MozUserSelect: "none",
                    msUserSelect: "none",
                    UserSelect: "none"
                }}>
                    <strong>Follow the box with the mouse cursor</strong>
                </p>
                <p style={{
                    WebkitUserSelect: "none",
                    MozUserSelect: "none",
                    msUserSelect: "none",
                    UserSelect: "none"
                }}>Task starts as soon as the mouse cursor is inside the box and will end after:&nbsp;
                    {!this.state.started ? this.props.phaseName.slice(0,2) === "PR" ? "20" : "30" : <Timer time={this.props.phaseName.slice(0,2) === "PR" ? 20 : 30} end={()=> this.endTask()}/>}</p>
                <div style={taskContainerStyle}>
                    <div className="movingBox"
                         ref={this.boxRef}
                         style={{backgroundColor: this.state.onObject ? "hsl(217, 71%, 53%)" : "hsl(0, 0%, 86%)",
                             animationPlayState: this.state.started ? "running" : "paused"}}

                    ></div>
                </div>
                {this.props.phaseName.slice(0,2) === "PR" ? this.renderInstructionModal() : null}
            </div>
        );
    }

}