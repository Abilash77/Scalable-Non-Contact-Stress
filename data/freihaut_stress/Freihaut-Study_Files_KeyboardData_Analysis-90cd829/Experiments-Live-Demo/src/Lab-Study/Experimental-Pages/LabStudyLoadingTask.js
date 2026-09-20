// Fake loader that is shown before the start of each condition to capture mouse and keyboard behavior during
// a non-goal directed (natural) setting
import React, { Component } from 'react';
import '../Style-Sheets/Lab-Loader.css';

import KeyboardMouseTracker from "../../General-Components/KeyboardMouseTracker";

const loaderStyle = {
    display: "flex",
    alignItems: "center",
    flexDirection: "column",
    justifyContent: "center",
    height: "550px"
};

export default class LoadingPage extends Component {

    componentDidMount() {
        window.scrollTo(0, 0);
        this.end = setTimeout(
            () => this.props.proceedPhase(this.props.phaseName, {mouseKeyboardData: this.mouseKeyboardData}),
            6000
        );

        this.mouseKeyboardData = [];
    }

    componentWillUnmount() {
        clearTimeout(this.end);
    }


    onKeyboardMouseEvent(datapoint) {
        // get tracker datapoint and add page relevant info before pushing it into the data array
        const pageInfo = {
            page: this.props.phaseName
        };

        Object.assign(datapoint, pageInfo);
        // console.log(datapoint);
        this.mouseKeyboardData.push(datapoint);
    }

    render() {
        return(
            <div>
                <KeyboardMouseTracker onEvent={this.onKeyboardMouseEvent.bind(this)}/>
                <p><span style={{visibility: "hidden"}}>Placeholder</span></p>
                <div style={{height: "550px",
                    width: "900px",
                    // border: "solid 3px black",
                    marginTop: "10px"}}>
                    <div className="content">
                        <div style={loaderStyle}>
                            <h5>The tasks will start any minute. Please wait.</h5>
                            <div className="loader"></div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}